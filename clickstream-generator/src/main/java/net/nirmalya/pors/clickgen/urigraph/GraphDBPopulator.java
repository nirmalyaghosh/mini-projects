package net.nirmalya.pors.clickgen.urigraph;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.factory.GraphDatabaseFactory;
import org.neo4j.graphdb.index.Index;
import org.neo4j.graphdb.index.IndexManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Populates a graph based on the indicated CSV file(s).
 * 
 * @author Nirmalya Ghosh
 */
public class GraphDBPopulator {

	private Logger logger = LoggerFactory.getLogger(GraphDBPopulator.class);

	private Node addNode(GraphDatabaseService graphDb, Index<Node> index, String propertyName, String propertyValue) {
		Node node = index.get(propertyName, propertyValue).getSingle();
		if (node == null) {
			logger.debug("Creating node with '{}'='{}'", propertyName, propertyValue);
			node = graphDb.createNode();
			node.setProperty(propertyName, propertyValue);
			index.add(node, propertyName, propertyValue);
		} else {
			logger.trace("Node with '{}'='{}' already exists", propertyName, propertyValue);
		}

		return node;
	}

	private void addRelationship(Node srcNode, Node dstNode, RelTypes relationshipType) {
		boolean foundRelationship = false;
		Iterable<Relationship> relationships = srcNode.getRelationships();
		if (relationships != null) {
			Iterator<Relationship> iterator = relationships.iterator();
			while (iterator.hasNext()) {
				Relationship relationship = iterator.next();
				Node startNode = relationship.getStartNode();
				Node endNode = relationship.getEndNode();
				if ((startNode != null) && (endNode != null)) {
					if (startNode.equals(srcNode) && endNode.equals(dstNode)) {
						logger.trace("Found relationship between {} and {}", srcNode, dstNode);
						foundRelationship = true;
					}
				}
			}
		}

		if (!foundRelationship) {
			logger.debug("Adding relationship between {} and {}", srcNode, dstNode);
			srcNode.createRelationshipTo(dstNode, relationshipType);
		}
	}

	public GraphDatabaseService populate(String graphDbPath, String... csvFilePaths) {

		logger.debug("Populating graphDB");
		long t1 = System.currentTimeMillis();

		GraphDatabaseService graphDb = new GraphDatabaseFactory().newEmbeddedDatabase(graphDbPath);
		IndexManager indexManager = graphDb.index();

		for (String csvFilePath : csvFilePaths) {
			List<String> lines = readFile(csvFilePath);
			Transaction tx = graphDb.beginTx();
			Index<Node> index = indexManager.forNodes("pageUri");
			try {
				for (String line : lines) {
					String[] columns = line.split(",");
					Node srcNode = addNode(graphDb, index, "pageUri", columns[0]);
					Node dstNode = addNode(graphDb, index, "pageUri", columns[1]);
					addRelationship(srcNode, dstNode, RelTypes.LINKS_TO);
				}
				tx.success();
			} catch (Exception e) {
				tx.failure();
			}
		}

		logger.debug("Finished populating graphDB. Time taken : {}", (System.currentTimeMillis() - t1));

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
			logger.debug("Caught an IOException whilst reading {} {}", csvFilePath, e);
		}

		return lines;
	}
}
