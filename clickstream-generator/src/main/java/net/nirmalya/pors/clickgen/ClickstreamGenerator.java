package net.nirmalya.pors.clickgen;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * @author Nirmalya Ghosh
 */
public class ClickstreamGenerator {

	private static Logger logger = LoggerFactory.getLogger(ClickstreamGenerator.class);

	private static String PROPERTIES_FILE_PATH = "path-to-properties-file";
	private static String URI_CSV_FILE_PATH = "path-to-uri-csv-files";
	private static String URI_GRAPH_FILE_PATH = "path-to-uri-graph-file";

	private static Options constructOptions() {
		Options options = new Options();

		options.addOption(null, URI_CSV_FILE_PATH, true, "path to one or more files containing comma separated "
				+ "list of URIs (one pair per line) " + "e.g. A,B (indicates B accessible from A");

		options.addOption(null, URI_GRAPH_FILE_PATH, true, "path to file storing the URI graph");

		options.addOption(null, PROPERTIES_FILE_PATH, true, "path to ClickstreamGenerator.properties file");

		return options;
	}

	public static void main(String[] args) throws ParseException {
		CommandLineParser parser = new BasicParser();
		Options options = constructOptions();

		// Parse the command line arguments
		CommandLine line = parser.parse(options, args);

		// Ensure that required options have been specified
		if (!line.hasOption(URI_GRAPH_FILE_PATH) || !line.hasOption(URI_CSV_FILE_PATH)
				|| !line.hasOption(PROPERTIES_FILE_PATH)) {
			printOptions(options);
			return;
		}

		ClickstreamGenerator generator = new ClickstreamGenerator();
		generator.execute(line);
	}

	private static void printOptions(Options options) {
		HelpFormatter helpFormatter = new HelpFormatter();
		helpFormatter.printHelp("java -jar <name-of-this-jar-file>.jar <options>", options);
	}

	public void execute(CommandLine line) {
		// Read the required command line arguments
		String graphDbPath = line.getOptionValue(URI_GRAPH_FILE_PATH);
		String[] csvFiles = line.getOptionValue(URI_CSV_FILE_PATH).split(",");

		// Start the URI sequence produce and consumer
		BlockingQueue<String> queue = new ArrayBlockingQueue<String>(2);
		new Thread(new URISequenceProducer(graphDbPath, csvFiles, queue)).start();
		new Thread(new URISequenceConsumer(queue)).start();

		// TODO Auto-generated method stub
		logger.debug("Shut down");
	}
}
