FROM mambaorg/micromamba:0.15.3
USER root
RUN mkdir /opt/chat_with_docs
RUN chmod -R 777 /opt/chat_with_docs
WORKDIR /opt/chat_with_docs
USER micromamba
COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
   micromamba clean --all --yes
COPY run.sh run.sh
COPY . .
USER root
RUN chmod a+x run.sh
CMD ["./run.sh"]
