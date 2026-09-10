FROM tensorflow/serving:latest

COPY serving_model/ /models/wine-quality/
COPY config/monitoring.config /config/monitoring.config
COPY tf_serving_entrypoint.sh /usr/bin/tf_serving_entrypoint.sh

ENV MODEL_NAME=wine-quality
ENV MODEL_BASE_PATH=/models/wine-quality
ENV MONITORING_CONFIG=/config/monitoring.config
ENV PORT=8501

EXPOSE 8501

RUN chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
