import ccxt
import pandas as pd
import time
from bitget_config import API_KEY, SECRET_KEY, PASSPHRASE

# ==========================================
# 設定
# ==========================================
SYMBOL      = 'BTC/USDT'   # 取引ペア
TIMEFRAME   = '1h'         # 足の種類
TRADE_AMOUNT = 0.001       # 1回の取引量（BTC）※約8,000円相当
RSI_PERIOD  = 14           # RSIの計算期間
MA_SHORT    = 25           # 短期移動平均
MA_LONG     = 75           # 長期移動平均
STOP_LOSS   = 0.02         # 損切りライン（2%）
CHECK_INTERVAL = 60 * 60  # チェック間隔（1時間）

# ==========================================
# BitGetに接続
# ==========================================
exchange = ccxt.bitget({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'password': PASSPHRASE,
    'options': {'defaultType': 'spot'},  # 現物取引
})

def get_data():
    """ローソク足データを取得してインジケーターを計算"""
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', '始値', '高値', '安値', '終値', '出来高'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 移動平均
    df['MA25'] = df['終値'].rolling(window=MA_SHORT).mean()
    df['MA75'] = df['終値'].rolling(window=MA_LONG).mean()

    # RSI
    delta = df['終値'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # クロス検出
    df['前のMA25'] = df['MA25'].shift(1)
    df['前のMA75'] = df['MA75'].shift(1)
    df['ゴールデンクロス'] = (df['MA25'] > df['MA75']) & (df['前のMA25'] <= df['前のMA75'])
    df['デッドクロス'] = (df['MA25'] < df['MA75']) & (df['前のMA25'] >= df['前のMA75'])

    return df

def get_position():
    """現在のBTC保有量を取得"""
    balance = exchange.fetch_balance()
    btc = balance['total'].get('BTC', 0)
    return btc

def buy():
    """成行で買い注文"""
    order = exchange.create_market_buy_order(SYMBOL, TRADE_AMOUNT)
    print(f"  → 買い注文完了: {order['id']}")
    return order

def sell(amount):
    """成行で売り注文"""
    order = exchange.create_market_sell_order(SYMBOL, amount)
    print(f"  → 売り注文完了: {order['id']}")
    return order

def run():
    """メインループ"""
    print("=" * 40)
    print("自動売買ボット起動")
    print(f"取引ペア: {SYMBOL}")
    print(f"取引量: {TRADE_AMOUNT} BTC")
    print("=" * 40)

    買値 = 0  # 買った価格を記録

    while True:
        try:
            now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            df = get_data()
            latest = df.iloc[-1]
            prev = df.iloc[-2]  # 1つ前の確定足を使う

            現在価格 = latest['終値']
            rsi = latest['RSI']
            ポジション = get_position()

            print(f"\n[{now}]")
            print(f"  現在価格: {現在価格:,.2f} USDT")
            print(f"  RSI: {rsi:.1f}")
            print(f"  MA25: {latest['MA25']:,.2f} | MA75: {latest['MA75']:,.2f}")
            print(f"  BTC保有量: {ポジション:.6f}")

            # ポジションなし → 買いシグナル確認
            if ポジション < 0.0005:
                if prev['ゴールデンクロス'] and rsi >= 50:
                    print("  ★ ゴールデンクロス＋RSI条件成立 → 買い！")
                    buy()
                    買値 = 現在価格
                else:
                    print("  → 買いシグナルなし、待機中")

            # ポジションあり → 売りシグナル確認
            else:
                # 損切り
                if 買値 > 0 and 現在価格 <= 買値 * (1 - STOP_LOSS):
                    print(f"  ✕ 損切りライン到達 → 売り！")
                    sell(ポジション)
                    買値 = 0

                # デッドクロス
                elif prev['デッドクロス']:
                    print(f"  ★ デッドクロス → 売り！")
                    sell(ポジション)
                    買値 = 0

                else:
                    損益率 = ((現在価格 - 買値) / 買値 * 100) if 買値 > 0 else 0
                    print(f"  → ポジション保有中 (損益: {損益率:+.2f}%)")

        except Exception as e:
            print(f"  エラー発生: {e}")

        # 1時間待機
        print(f"\n  次のチェックまで1時間待機...")
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    run()