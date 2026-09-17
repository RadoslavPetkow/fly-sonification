from dotenv import load_dotenv; load_dotenv()
from neuprint import Client
c = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0")
print(c.fetch_version())
print(c.fetch_datasets().keys())
