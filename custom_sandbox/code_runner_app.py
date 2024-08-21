from flask import Flask, request
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

app = Flask(__name__)
API_KEY = os.environ.get('SANDBOX_API_KEY') or "dify-sandbox"


def validate_api_key():
    key = request.headers.get('X-Api-Key')
    if key != API_KEY:
        raise ValueError('api key not correct')


@app.route("/v1/sandbox/dependencies")
def installed_dependencies():
    validate_api_key()
    deps = {
        "code": 0,
        "message": "success",
        "data": {
            "dependencies": []
        }
    }
    return deps


@app.route("/v1/sandbox/run", methods=['POST'])
def run_code():
    validate_api_key()
    req = request.get_json(force=True)
    logging.info("Got code execution request: {}".format(req))
    code = req['code']
    logging.info(f"Code: {code}")
    try:
        lang = req['language']
        if lang != 'python3':
            raise ValueError('Only python is supported')

        start = time.time()
        loc = {}
        exec(code, {}, loc)  # globals()
        result = loc['result']
        elapsed = time.time() - start
        logging.info(f'execution elapsed: {elapsed}s, result: {result}')

        result_wrapped = {
            "code": 0,
            "message": "success",
            "data": {
                "error": "",
                "stdout": result
            }
        }
    except Exception as e:
        logging.exception("Failed to execute code", e)
        result_wrapped = {
            "code": 0,
            "message": "failure",
            "data": {
                "error": str(e),
                "stdout": ""
            }
        }

    return result_wrapped


if __name__ == '__main__':
    app.run('0.0.0.0', 5002, debug=True)
