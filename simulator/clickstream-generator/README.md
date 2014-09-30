##Clickstream Generator

Simulates user clickstream based on a specified page navigation graph.
+ Includes a random but realistic IP address generator
+ Underlying graph database (Neo4J) used to simulate pseudo-random sequence of page visits (traversals)
+ Generates 'sessionIDs' to simulate individual browsing sessions

``` bash
# Build the single executatable JAR file
mvn clean compile assembly:single

# Clear the Neo4J DB
rm -rf ~/tmp/neo4j-db/uri-graphdb

# Run
./scripts/run-clickgen.sh
```
Refer to the __scripts__ folder