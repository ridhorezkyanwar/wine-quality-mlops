FROM prom/prometheus:v3.15.1

COPY prometheus.yml /etc/prometheus/prometheus.yml
COPY prometheus.local.yml /etc/prometheus/prometheus.local.yml

EXPOSE 9090

CMD ["--config.file=/etc/prometheus/prometheus.yml", \
     "--storage.tsdb.path=/prometheus", \
     "--web.enable-lifecycle"]
