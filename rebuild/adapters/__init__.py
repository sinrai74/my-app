"""
adapters層（Step5-1）: Legacyデータ取得をrebuildから利用するためのProvider実装。

責務（Step5-1指示）:
  - Legacyデータ取得（既存fetch/変換関数をimportして呼ぶ・二重実装禁止）
  - Step1モデル / Ver4互換boatsへの変換
  - Provider専用例外への変換
  - 開始/終了/件数/実行時間のログのみ

禁止: 評価・スコア・buyscore・EV・Kelly・判定・保存・出力・通知。

依存規則: adapters → models（Coreモデル）のみ。
  Storage / Output / Notification / Pipeline へは依存しない。
  BoatsResolverはVer4Engine互換のための例外レイヤーであり、Provider内部に閉じる。
"""

from adapters.exceptions import ProviderError
from adapters.providers import (
    BoatsProvider,
    BoatsResolver,
    OddsProvider,
    RaceProvider,
    RaceTypeProvider,
    VenueProvider,
    WeatherProvider,
)

__all__ = [
    "BoatsProvider",
    "BoatsResolver",
    "OddsProvider",
    "ProviderError",
    "RaceProvider",
    "RaceTypeProvider",
    "VenueProvider",
    "WeatherProvider",
]
