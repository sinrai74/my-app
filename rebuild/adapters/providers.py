"""
Provider実装（Step5-1）: Legacy取得 → Coreモデル / Ver4互換boats 変換。

方針（Step5-1レビュー確定）:
  取得は1回（BoatInfo等を共通ソースとして取得）、変換は2種類:
    - resolve_race(...)  → Race（RaceEntry/Weather/OddsSnapshot/RaceType込み）
    - resolve_boats(...) → Ver4互換boats（BOAT_ATTRS Mapping列）
  BoatsResolverはVer4Engine互換のための例外レイヤーであり、Provider内部に閉じる。
  RaceEntryへ不足属性を追加してはならない（Step1モデル・Feature Freeze変更禁止）。

責務: Legacy取得・モデル変換・Provider例外変換・最小ログのみ。
  評価/スコア/buyscore/EV/Kelly/判定/保存/出力/通知はしない。

依存: adapters → models のみ。Legacy本番モジュールはimportして呼ぶ（無改変）。
  本番モジュール（notify_arashi等）はimportコストと副作用回避のため、
  各メソッド内で遅延importする（テストではDIしたfetcherで差し替え可能）。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from adapters.exceptions import ProviderError
from models.race import (
    CourseStats,
    OddsSnapshot,
    Race,
    RaceEntry,
    Weather,
)

log = logging.getLogger(__name__)

# Ver4Engine（core.engine）が要求するboat属性。core側の定義と一致させる
# （core.engine.golden_wrapper BOAT_ATTRSと同一。ここはVer4互換レイヤー）。
BOAT_ATTRS: tuple[str, ...] = (
    "lane", "name", "racer_class", "win_rate", "avg_st", "motor", "local_win",
    "ability_curr", "ability_prev", "course_nyuko", "course_win_rate",
    "course_place_rate", "course_place_counts", "course_rank", "course_st",
    "course_f_count", "course_l_count",
)

NIGHT_VENUES = frozenset({4, 6, 12, 17, 20, 21, 22, 23, 24})


# ==================== Protocol群 ====================


class RaceProvider(Protocol):
    """レース識別子 → Race（ドメインモデル）。

    入力: race_date(YYYYMMDD), venue_num(1-24), race_number(1-12)
    出力: Race（RaceEntry列・Weather・grade・is_night等のドメインモデル）
    責務: 出走表を取得し、Step1のドメインモデルRaceへ変換する
    禁止: 評価・スコア・buyscore・EV・Kelly・判定・保存・出力・通知。
          Ver4互換boats（詳細17属性）の生成はしない（それはBoatsResolver）。

    RaceProvider と BoatsResolver の責務境界:
      両者は同じ出走表ソースを使うが、用途と出力が異なる。
      - RaceProvider  : Pipeline/ドメイン用の「軽量な」Race（RaceEntry）。
                        Step1で確定した正式モデル。Feature Freeze対象。
      - BoatsResolver : Ver4Engine互換の「詳細な」boats（BOAT_ATTRS 17属性）。
                        RaceEntryに無いlocal_win/ability_*/course_*を含む、
                        Ver4評価専用の互換アダプタ出力（Coreモデルではない）。
      同一Provider（BoatsProvider）が両方を実装するが、RaceEntryへVer4属性を
      追加して一本化することは禁止（Step1モデル変更＝Freeze違反になるため）。
    """

    def resolve_race(
        self, race_date: str, venue_num: int, race_number: int
    ) -> Race: ...


class BoatsResolver(Protocol):
    """レース識別子 → Ver4互換boats（BOAT_ATTRS Mapping列・Ver4Engine専用）。

    入力: race_date, venue_num, race_number
    出力: Sequence[Mapping[str,Any]]（各艇のBOAT_ATTRS 17属性dict）
    責務: 出走表からVer4Engineが要求する詳細属性を投影する互換アダプタ
    禁止: 評価・判定・保存・出力・通知。Race/RaceEntryの生成（それはRaceProvider）。

    位置づけ: これはCoreモデルではなくVer4互換維持のための例外レイヤーであり、
      Provider内部に閉じる。Coreモデル（RaceEntry等）へ影響を与えてはならない。
    """

    def resolve_boats(
        self, race_date: str, venue_num: int, race_number: int
    ) -> Sequence[Mapping[str, Any]]: ...


class OddsProvider(Protocol):
    """3連単オッズ取得。

    入力: race_date, venue_num, race_number
    出力: OddsSnapshot（trifecta_odds: {"1-2-3": 47.2, ...}）
    責務: 公式サイト等からオッズを取得しOddsSnapshotへ変換する
    禁止: EV計算・期待値判定・buyscore・Kelly・買い目選定・保存・通知。
    """

    def resolve_odds(
        self, race_date: str, venue_num: int, race_number: int
    ) -> OddsSnapshot: ...


class VenueProvider(Protocol):
    """会場情報取得。

    入力: venue_num(1-24)
    出力: 場名(str)
    責務: 会場番号から会場名を解決する（静的マスタ参照）
    禁止: 統計補正・水面タイプ分類・場×コース補正（それらはVenueStatsProvider
          =Step4のcore側Providerの責務。本Providerは会場名取得のみ）。
    """

    def resolve_venue_name(self, venue_num: int) -> str: ...


class WeatherProvider(Protocol):
    """気象取得（風/波/天候/気温）。

    入力: race_date, venue_num, race_number
    出力: Optional[Weather]（データ無しはNone）
    責務: 気象データを取得しWeatherモデルへ変換する
    禁止: 気象を用いた評価・補正・判定（Weatherは朝刊のみポリシーで評価に
          使われない。Providerは取得と変換のみ）。
    """

    def resolve_weather(
        self, race_date: str, venue_num: int, race_number: int
    ) -> Optional[Weather]: ...


class RaceTypeProvider(Protocol):
    """race_type取得。

    入力: race_date, venue_num, race_number（＋分類に要する素材）
    出力: race_type文字列（例 "イン逃げ型"/"混戦型" 等、Legacy分類の戻り値）
    責務: Legacyの分類関数を呼び、race_type文字列を返すのみ
    禁止: _race_type_bonus計算・investment_type生成・buyscore・判定。
          分類ロジック自体の再実装（Legacy関数をimportして呼ぶだけ）。

    注記（Step5-1調査結果）: Legacyの分類戻り値（"イン逃げ型"等）と、
      buyscoreの_race_type_bonusが参照するキー（"本命戦"/"混戦"等）は
      体系が異なる。この差の解消はProviderの責務ではなく、Pipeline結線
      （Step5-2）で扱う。Providerは取得値をそのまま返す。
    """

    def resolve_race_type(
        self, race_date: str, venue_num: int, race_number: int
    ) -> str: ...


# 依存注入用のフェッチャ型（テストで差し替え可能にする）
ProgramsFetcher = Callable[[str], list[dict]]
OddsFetcher = Callable[[int, str, str], Mapping[str, float]]
BoatsExtractor = Callable[[dict], list]
RaceTypeClassifier = Callable[..., str]


# ==================== VenueProvider ====================

# 会場番号 → 場名（notify_arashi / x_asahi_scoring の VENUE_NAMES と同一値）。
# 会場マスタは静的定数であり「取得」の一形態。Legacyと同一表を用いる。
_VENUE_NAMES: dict[int, str] = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川",
    6: "浜名湖", 7: "蒲郡", 8: "常滑", 9: "津", 10: "三国",
    11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀",
    16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松",
    21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}


class DefaultVenueProvider:
    """会場名の解決（静的マスタ参照のみ）。"""

    def resolve_venue_name(self, venue_num: int) -> str:
        name = _VENUE_NAMES.get(venue_num)
        if name is None:
            raise ProviderError(f"unknown venue_num: {venue_num}")
        return name


# ==================== OddsProvider ====================


class DefaultOddsProvider:
    """odds_fetch.fetch_odds をラップしてOddsSnapshotへ変換する。"""

    def __init__(self, odds_fetcher: Optional[OddsFetcher] = None) -> None:
        self._fetch = odds_fetcher

    def resolve_odds(
        self, race_date: str, venue_num: int, race_number: int
    ) -> OddsSnapshot:
        start = time.monotonic()
        log.info(
            "OddsProvider start date=%s venue=%s race=%s",
            race_date, venue_num, race_number,
        )
        fetch = self._fetch
        if fetch is None:
            from odds_fetch import fetch_odds as fetch  # 遅延import・無改変
        try:
            odds_map = fetch(race_number, f"{venue_num:02d}", race_date)
        except Exception as exc:  # 取得失敗はProviderErrorへ正規化
            raise ProviderError(
                f"odds fetch failed for {race_date}_{venue_num:02d}_{race_number:02d}"
            ) from exc

        eval_id = f"{race_date}_{venue_num:02d}_{race_number:02d}"
        snapshot = OddsSnapshot(
            eval_id=eval_id,
            fetched_at=_now_iso(),
            trifecta_odds=dict(odds_map),
        )
        log.info(
            "OddsProvider end count=%d elapsed=%.3fs",
            len(snapshot.trifecta_odds), time.monotonic() - start,
        )
        return snapshot


# ==================== BoatsProvider（取得1回・変換2種） ====================


class BoatsProvider:
    """Legacy出走表を1回取得し、Race変換とVer4 boats変換の2種を提供する。

    取得: notify_arashi.fetch_programs → _extract_boats_from_program（無改変）。
    変換1: resolve_race  → Race（RaceEntry/Weather/OddsSnapshot/RaceType込み）
    変換2: resolve_boats → Ver4互換boats（BOAT_ATTRS Mapping列）

    二重取得を避けるため、同一(race_date,venue,race)のprogram/boat_objsを
    インスタンス内でキャッシュする（純粋な取得結果のメモ化のみ。判定はしない）。
    """

    def __init__(
        self,
        programs_fetcher: Optional[ProgramsFetcher] = None,
        boats_extractor: Optional[BoatsExtractor] = None,
        venue_provider: Optional[VenueProvider] = None,
        weather_provider: Optional[WeatherProvider] = None,
        odds_provider: Optional[OddsProvider] = None,
        race_type_provider: Optional[RaceTypeProvider] = None,
    ) -> None:
        self._programs_fetcher = programs_fetcher
        self._boats_extractor = boats_extractor
        self._venue = venue_provider or DefaultVenueProvider()
        self._weather = weather_provider
        self._odds = odds_provider
        self._race_type = race_type_provider
        # (race_date, venue_num, race_number) -> (program, boat_objs)
        self._cache: dict[tuple[str, int, int], tuple[dict, list]] = {}

    # ---- 共通取得（1回だけ・メモ化） ----

    def _get_source(
        self, race_date: str, venue_num: int, race_number: int
    ) -> tuple[dict, list]:
        key = (race_date, venue_num, race_number)
        if key in self._cache:
            return self._cache[key]

        start = time.monotonic()
        log.info(
            "BoatsProvider fetch start date=%s venue=%s race=%s",
            race_date, venue_num, race_number,
        )
        fetch_programs = self._programs_fetcher
        extract = self._boats_extractor
        if fetch_programs is None or extract is None:
            from notify_arashi import (  # 遅延import・無改変
                _extract_boats_from_program as _legacy_extract,
                fetch_programs as _legacy_fetch,
            )
            fetch_programs = fetch_programs or _legacy_fetch
            extract = extract or _legacy_extract

        try:
            programs = fetch_programs(race_date)
        except Exception as exc:
            raise ProviderError(f"programs fetch failed for {race_date}") from exc
        if not programs:
            raise ProviderError(f"no programs returned for {race_date}")

        program = next(
            (
                p for p in programs
                if int(p.get("race_stadium_number", 0)) == venue_num
                and int(p.get("race_number", 0)) == race_number
            ),
            None,
        )
        if program is None:
            raise ProviderError(
                f"program not found: {race_date}_{venue_num:02d}_{race_number:02d}"
            )

        try:
            boat_objs = extract(program)
        except Exception as exc:
            raise ProviderError(
                f"boat extraction failed: {race_date}_{venue_num:02d}_{race_number:02d}"
            ) from exc
        if not boat_objs:
            raise ProviderError(
                f"no boats extracted: {race_date}_{venue_num:02d}_{race_number:02d}"
            )

        self._cache[key] = (program, boat_objs)
        log.info(
            "BoatsProvider fetch end count=%d elapsed=%.3fs",
            len(boat_objs), time.monotonic() - start,
        )
        return program, boat_objs

    # ---- 変換1: Race（ドメインモデル） ----

    def resolve_race(
        self, race_date: str, venue_num: int, race_number: int
    ) -> Race:
        program, boat_objs = self._get_source(race_date, venue_num, race_number)
        entries = tuple(_boat_to_race_entry(b) for b in boat_objs)

        weather = None
        if self._weather is not None:
            weather = self._weather.resolve_weather(race_date, venue_num, race_number)

        venue_name = self._venue.resolve_venue_name(venue_num)
        grade = _grade_number_to_str(int(program.get("race_grade_number", 0) or 0))

        return Race(
            race_date=race_date,
            venue_num=venue_num,
            venue_name=venue_name,
            race_number=race_number,
            close_time=str(program.get("race_closed_at", "") or ""),
            is_night=venue_num in NIGHT_VENUES,
            entries=entries,
            grade=grade,
            weather=weather,
        )

    # ---- 変換2: Ver4互換boats ----

    def resolve_boats(
        self, race_date: str, venue_num: int, race_number: int
    ) -> Sequence[Mapping[str, Any]]:
        _program, boat_objs = self._get_source(race_date, venue_num, race_number)
        return [
            {attr: getattr(b, attr, None) for attr in BOAT_ATTRS} for b in boat_objs
        ]


# ==================== WeatherProvider ====================


class DefaultWeatherProvider:
    """program由来の気象をWeatherへ変換する（取得のみ）。

    weather_fetcher(race_date, venue_num, race_number) -> dict を注入する。
    dict想定キー: wind_speed / wind_direction / wave_height / temperature /
    water_temperature（notify_arashiの気象dictと同一）。
    """

    def __init__(
        self,
        weather_fetcher: Callable[[str, int, int], Optional[Mapping[str, Any]]],
    ) -> None:
        self._fetch = weather_fetcher

    def resolve_weather(
        self, race_date: str, venue_num: int, race_number: int
    ) -> Optional[Weather]:
        try:
            data = self._fetch(race_date, venue_num, race_number)
        except Exception as exc:
            raise ProviderError(
                f"weather fetch failed: {race_date}_{venue_num:02d}_{race_number:02d}"
            ) from exc
        if not data:
            return None
        wave = data.get("wave_height")
        return Weather(
            wind_speed_mps=float(data.get("wind_speed") or 0.0),
            wind_direction=str(data.get("wind_direction", "") or ""),
            wave_height_cm=int(wave) if wave is not None else 0,
            temperature_celsius=_opt_float(data.get("temperature")),
            water_temperature_celsius=_opt_float(data.get("water_temperature")),
        )


# ==================== RaceTypeProvider ====================


class DefaultRaceTypeProvider:
    """notify_arashi._classify_race_type をラップする（取得のみ・_bonus禁止）。

    classifier は race_type文字列を返すCallable。分類に必要な引数は
    呼び出し側（Pipeline=Step5-2）が渡す設計だが、Step5-1では
    「分類関数を呼ぶだけ」の薄いアダプタとして境界を確定する。
    分類入力の結線はStep5-2で行うため、ここでは classifier と
    その引数を受け取る形にとどめる（新規ロジック追加なし）。
    """

    def __init__(self, classifier: Optional[RaceTypeClassifier] = None) -> None:
        self._classifier = classifier

    def resolve_race_type(
        self, race_date: str, venue_num: int, race_number: int, **kwargs: Any
    ) -> str:
        classifier = self._classifier
        if classifier is None:
            from notify_arashi import _classify_race_type as classifier  # 遅延import
        try:
            return classifier(**kwargs)
        except Exception as exc:
            raise ProviderError(
                f"race_type classify failed: "
                f"{race_date}_{venue_num:02d}_{race_number:02d}"
            ) from exc


# ==================== 内部ヘルパー（純粋変換） ====================


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _grade_number_to_str(n: int) -> str:
    """race_grade_number → Race.grade文字列（Ver4Engine._grade_to_numberの逆）。

    0-4は一般/G3/G2/G1/SG、未知は"gradeN"（Ver4Engineが往復可能な表現）。
    """
    mapping = {0: "一般", 1: "G3", 2: "G2", 3: "G1", 4: "SG"}
    return mapping.get(n, f"grade{n}")


def _boat_to_race_entry(boat: Any) -> RaceEntry:
    """BoatInfo → RaceEntry（Step1モデル）。

    RaceEntryが持つフィールドのみを変換する。Ver4評価専用の詳細属性
    （local_win/ability_*/course_*）はRaceEntryに追加しない（Freeze厳守）。
    course_stats はBoatInfoのコース別配列から CourseStats へ再構成する。
    """
    course_stats = _build_course_stats(boat)
    return RaceEntry(
        lane=getattr(boat, "lane"),
        racer_no=str(getattr(boat, "racer_id", "") or ""),
        racer_name=str(getattr(boat, "name", "") or ""),
        racer_class=str(getattr(boat, "racer_class", "") or ""),
        win_rate=float(getattr(boat, "win_rate", 0.0) or 0.0),
        place_rate=0.0,  # BoatInfoに全国複勝率フィールドが無いため0.0
        motor_no=0,  # BoatInfoにモーター番号フィールドが無いため0
        motor_rate2=float(getattr(boat, "motor", 0.0) or 0.0),
        avg_st=float(getattr(boat, "avg_st", 0.0) or 0.0),
        course_stats=course_stats,
    )


def _build_course_stats(boat: Any) -> Optional[tuple[CourseStats, ...]]:
    """BoatInfoのコース別配列 → CourseStats列（存在する範囲のみ）。"""
    nyuko = getattr(boat, "course_nyuko", None)
    place_rate = getattr(boat, "course_place_rate", None)
    course_st = getattr(boat, "course_st", None)
    course_rank = getattr(boat, "course_rank", None)
    if not nyuko:
        return None
    stats: list[CourseStats] = []
    for i in range(min(6, len(nyuko))):
        stats.append(
            CourseStats(
                course=i + 1,
                entry_count=int(nyuko[i]),
                place_rate=float(place_rate[i]) if place_rate else 0.0,
                avg_start_timing=float(course_st[i]) if course_st else 0.0,
                avg_start_rank=float(course_rank[i]) if course_rank else 0.0,
            )
        )
    return tuple(stats)
