from roostoo import RoostooClient


client = RoostooClient()

print("\nSERVER TIME")
print(client.server_time())

print("\nSYNC TIME")
print(client.sync_time())

print("\nEXCHANGE INFO")
info = client.exchange_info()

print("Running:", info["IsRunning"])
print("Pairs:", len(info["TradePairs"]))

print("\nTICKER")
ticker = client.ticker()

print("Pairs returned:", len(ticker["Data"]))

print("\nBTC")
print(ticker["Data"].get("BTC/USD"))

print("\nBALANCE")
print(client.balance())

print("\nPENDING ORDERS")
print(client.pending_count())