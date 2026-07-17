"""
プラットフォーム共通例外（Step2最小導入）。

設計書 Phase0.5 v1.1.3 ⑫（エラー設計）に基づく。
⑫の6分類（DataFetchError / ParseError / StorageError / DeliveryError /
ConfigError / ModelError）のうち、Step2ではMapper層が必要とする
ParseError と StorageError を導入する。残る4分類は必要とするStepで追加する。

注記（設計上の未決事項）: ⑫の例外はcore層も送出しうるが、⑩の依存ルール上
coreはstorageをimportできない。例外モジュールの最終的な配置
（全レイヤーから参照可能な場所への移設）は、coreが例外を必要とする
Stepで提案・承認のうえ決定する。Step2時点ではMapper専用としてここに置く。
"""

from __future__ import annotations


class PlatformError(Exception):
    """プラットフォーム共通の基底例外（設計書⑫）。"""


class StorageError(PlatformError):
    """永続化操作の失敗（一意性違反、ヘッダー不整合、書込先の状態異常など）。

    Step2-6で導入。データの「解釈」の失敗はParseError、保存先の「状態」の
    問題はStorageErrorと使い分ける。
    """


class ParseError(PlatformError):
    """データの解釈・変換の失敗（必須列の欠落、JSON列の破損、型の不整合など）。

    サイレント失敗の全面禁止（設計書⑫）に基づき、Mapperは不正な入力を
    黙って補正せず、本例外で明示的に失敗させる。
    """
