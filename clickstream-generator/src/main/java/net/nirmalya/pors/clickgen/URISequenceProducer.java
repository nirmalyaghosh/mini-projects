package net.nirmalya.pors.clickgen;

import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.BlockingQueue;

import net.nirmalya.pors.clickgen.urigraph.GraphDBPopulator;

import org.neo4j.cypher.javacompat.ExecutionEngine;
import org.neo4j.cypher.javacompat.ExecutionResult;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.ResourceIterator;
import org.neo4j.graphdb.Transaction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Generates URI sequences by first populating a graph DB and then iterating through the list of all paths.
 * <p>
 * This class adds the URI sequences to a <code>BlockingQueue</code> to be picked up by one or more
 * {@link URISequenceConsumer}s.
 * <p>
 * Sample URI sequence : <code>[page1, page3, page4, page5]</code>
 * 
 * @author Nirmalya Ghosh
 */
public class URISequenceProducer implements Runnable {

	private BlockingQueue<String> queue = null;
	private static Logger logger = LoggerFactory.getLogger(URISequenceProducer.class);
	private String graphDbPath;
	private String[] csvFiles;

	public URISequenceProducer(String graphDbPath, String[] csvFiles, BlockingQueue<String> queue) {
		this.graphDbPath = graphDbPath;
		this.csvFiles = csvFiles;
		this.queue = queue;
	}

	@Override
	public void run() {
		logger.debug("Started");

		// Populate the graph DB
		GraphDBPopulator populator = new GraphDBPopulator();
		GraphDatabaseService graphDb = populator.populate(this.graphDbPath, this.csvFiles);

		// Use the graph DB to get list of paths
		ExecutionEngine cypherQueryEngine = new ExecutionEngine(graphDb);
		ExecutionResult result;
		try (Transaction ignored = graphDb.beginTx()) {
			result = cypherQueryEngine.execute("START a = node(*) MATCH path = a-[r:LINKS_TO*]->b "
					+ "RETURN EXTRACT(n IN NODES(path)| n.pageUri)");

			ResourceIterator<Map<String, Object>> iterator = result.iterator();
			while (iterator.hasNext()) {
				Map<String, Object> row = iterator.next();
				for (Entry<String, Object> column : row.entrySet()) {
					String nodeSeqStr = column.getValue().toString();
					logger.debug("Sequence : {}", nodeSeqStr);
					try {
						this.queue.put(nodeSeqStr);
					} catch (InterruptedException e) {
						logger.debug("{}", e.getMessage());
					}
				}
			}

			logger.debug("Finished reading list of paths");
			try {
				queue.put("Shutdown");
			} catch (InterruptedException e) {
				logger.debug("{}", e.getMessage());
			}
		} catch (Exception e) {
			logger.debug("{}", e.getMessage());
		}

		logger.debug("Shut down");
	}

}
