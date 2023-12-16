import cProfile
import pstats
from memory import Memory
import torch

# メモリのインスタンスを作成
batch_size = 128
capacity = 1000000
state_shape = (30, 5)
trading_state_shape = (30, 5)
action_shape = 1
action_type = torch.int32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

memory = Memory(capacity, state_shape, trading_state_shape, action_shape, action_type, device)

# プロファイラを作成
profiler = cProfile.Profile()

# プロファイリングを開始
profiler.enable()

# プロファイリングしたいコードを実行
# ここでは、appendメソッドとsampleメソッドを何度か呼び出す例を示します。
# 実際には、プロファイリングしたい具体的なコードを記述してください。
for _ in range(10000):
    state = torch.randn(state_shape)
    trading_state = torch.randn(trading_state_shape)
    action = torch.tensor(1, dtype=action_type)
    reward = torch.randn(1)
    new_state = torch.randn(state_shape)
    new_trading_state = torch.randn(trading_state_shape)
    done = torch.tensor(1, dtype=torch.float32)

    memory.append(state, trading_state, action, reward, new_state, new_trading_state, done)

# プロファイリングを終了
profiler.disable()

# 結果を表示
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
