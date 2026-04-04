import time
import sys
import traceback
sys.path.append('05_dashboard')
from utils.data_loader import _load_bronze_filtered

start = time.time()
print("Starting _load_bronze_filtered...")
try:
    df = _load_bronze_filtered(2)
    print("Success! DataFrame size:", len(df))
    print("Time taken:", time.time() - start)
except Exception as e:
    print("EXCEPTION CAUGHT after", time.time() - start, "seconds:")
    print(type(e).__name__, ":", str(e).encode('ascii', 'ignore').decode('ascii'))
