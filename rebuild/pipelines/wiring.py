"""
Engine向けDI配線Adapter（Step5-3実装/Step5-4移設）: 引数の詰め替えのみ。

配置: pipelines/wiring.py（DI配線専用。入力Providerではないため
adapters/からpipelines/へ移設。処理内容はStep5-3から一切変更していない）。

Ver4Engineのboats_resolverは Callable[[Race], Sequence[Boat]]（Race引数）だが、
Step5-1 BoatsProvider.resolve_boats は (race_date, venue_num, race_number) 引数。
このシグネチャ差をDI境界で吸収する。

Adapterの責務（Step5-3指示）: 引数を詰め替えるだけ。
禁止: 判定・計算・補正・キャッシュ追加・データ加工。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from models.race import Race


class RaceArgBoatsResolver:
    """resolve_boats(date,venue,race) を Callable[[Race], boats] へ適合させる。

    Ver4Engineのboats_resolver（Race引数）として注入できるようにする。
    引数の詰め替えのみ。取得の実体・メモ化はBoatsProvider側が持つ
    （ここでキャッシュ・加工はしない）。
    """

    def __init__(self, provider: Any) -> None:
        # provider は BoatsResolver Protocol（resolve_boats を持つ具象）
        self._provider = provider

    def __call__(self, race: Race) -> Sequence[Mapping[str, Any]]:
        return self._provider.resolve_boats(
            race.race_date, race.venue_num, race.race_number
        )
