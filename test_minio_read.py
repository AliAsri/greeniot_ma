import traceback
from deltalake import DeltaTable
import sys

# Change console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

storage_options = {
    'AWS_ACCESS_KEY_ID': 'greeniot',
    'AWS_SECRET_ACCESS_KEY': 'greeniot2030',
    'AWS_ENDPOINT_URL': 'http://localhost:9000',
    'AWS_REGION': 'us-east-1',
    'AWS_ALLOW_HTTP': 'true',
    'AWS_S3_ALLOW_UNSAFE_RENAME': 'true'
}

print('Trying to connect to s3://greeniot/bronze/servers...')
try:
    dt = DeltaTable('s3://greeniot/bronze/servers', storage_options=storage_options)
    print("Success! Reading data...")
    arrow_table = dt.to_pyarrow_table()
    df = arrow_table.to_pandas()
    print("DataFrame size:", len(df))
except Exception as e:
    print('Error caught:')
    traceback.print_exc()
