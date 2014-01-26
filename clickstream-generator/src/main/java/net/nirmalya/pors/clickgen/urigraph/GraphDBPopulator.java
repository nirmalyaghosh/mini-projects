package net.nirmalya.pors.clickgen.urigraph;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.factory.GraphDatabaseFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Populates a graph based on the indicated CSV file(s).
 * 
 * @author Nirmalya Ghosh
 */
public class GraphDBPopulator {

	private Logger logger = LoggerFactory.getLogger(GraphDBPopulator.class);

	public GraphDatabaseService populate(String graphDbPath,
			String... csvFilePaths) {
		GraphDatabaseService graphDb = new GraphDatabaseFactory()
				.newEmbeddedDatabase(graphDbPath);

		for (String csvFilePath : csvFilePaths) {
			List<String> lines = readFile(csvFilePath);

			Transaction tx = graphDb.beginTx();
			try {
				for (String line : lines) {
					String[] columns = line.split(",");
					Node srcNode = graphDb.createNode();
					srcNode.setProperty("name", columns[0]);
					Node dstNode = graphDb.createNode();
					dstNode.setProperty("name", columns[1]);
					// TODO ideally we'd like to check if node already exists
				}
				tx.success();
			} finally {
				tx.finish();
			}
		}

		return graphDb;
	}

	private List<String> readFile(String csvFilePath) {
		logger.debug("Reading {}", csvFilePath);
		List<String> lines = new LinkedList<String>();
		Path path = Paths.get(csvFilePath);
		try (Scanner scanner = new Scanner(path, StandardCharsets.UTF_8.name())) {
			while (scanner.hasNextLine()) {
				lines.add(scanner.nextLine());
			}
		} catch (IOException e) {
			logger.debug("Caught an IOException whilst reading {}", csvFilePath);
		}

		return lines;
	}
}
