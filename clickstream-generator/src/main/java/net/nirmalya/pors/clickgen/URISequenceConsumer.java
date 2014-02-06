package net.nirmalya.pors.clickgen;

import java.util.concurrent.BlockingQueue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Consumes the sequence of URIs produced by the {@link URISequenceProducer}.
 * 
 * @author Nirmalya Ghosh
 */
public class URISequenceConsumer implements Runnable {

	private BlockingQueue<String> queue = null;
	private static Logger logger = LoggerFactory.getLogger(URISequenceConsumer.class);

	public URISequenceConsumer(BlockingQueue<String> queue) {
		this.queue = queue;
	}

	@Override
	public void run() {
		logger.debug("Started");
		final String SHUTDOWN_REQ = "Shutdown";
		String item;
		try {
			while ((item = (String) this.queue.take()) != SHUTDOWN_REQ) {
				processURISequence(item);
			}
			logger.debug("Shutting down");
		} catch (InterruptedException e) {
			logger.debug("{}", e.getMessage());
		}

		logger.debug("Shut down");
	}

	private void processURISequence(String item) throws InterruptedException {
		logger.debug("Processing sequence : {}", item);

		Thread.sleep(1000);
		// TODO requires implementation

	}

}
