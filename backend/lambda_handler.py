from mangum import Mangum
from app_fgsm import app

# AWS Lambda entrypoint
handler = Mangum(app)


