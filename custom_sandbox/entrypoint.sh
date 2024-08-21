if [[ "${DEBUG}" == "true" ]]; then
  exec flask run --host=${SANDBOX_BIND_ADDRESS:-0.0.0.0} --port=${SANDBOX_PORT:-8194} --debug
else
  exec gunicorn \
    --bind "${SANDBOX_BIND_ADDRESS:-0.0.0.0}:${SANDBOX_PORT:-8194}" \
    --workers ${SANDBOX_WORKER_AMOUNT:-1} \
    --worker-class ${SANDBOX_WORKER_CLASS:-gevent} \
    --timeout ${GUNICORN_TIMEOUT:-200} \
    --preload \
    code_runner_app:app
fi