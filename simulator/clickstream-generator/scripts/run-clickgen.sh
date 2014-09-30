java -jar ../target/clickstream-generator-0.1-SNAPSHOT-jar-with-dependencies.jar \
    --path-to-properties-file ../src/test/resources/ClickstreamGenerator.properties \
    --path-to-uri-csv-files ../src/test/resources/uri-graph-1.csv \
    --path-to-uri-graph-file ~/tmp/neo4j-db/uri-graphdb