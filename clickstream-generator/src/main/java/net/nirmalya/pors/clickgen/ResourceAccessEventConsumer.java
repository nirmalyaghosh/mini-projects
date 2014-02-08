package net.nirmalya.pors.clickgen;

import java.util.concurrent.BlockingQueue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Consumes the evens produced by the {@link URISequenceConsumer} and transforms the events into a specific log file
 * format.
 * 
 * @author Nirmalya Ghosh
 */
public class ResourceAccessEventConsumer implements Runnable {

	private BlockingQueue<ResourceAccessEvent> accessEventQueue = null;
	private static Logger logger = LoggerFactory.getLogger(ResourceAccessEventConsumer.class);

	public ResourceAccessEventConsumer(BlockingQueue<ResourceAccessEvent> accessEventQueue) {
		this.accessEventQueue = accessEventQueue;
	}

	@Override
	public void run() {
		logger.debug("Started");
		ResourceAccessEvent event;
		try {
			Thread.sleep(15000);
			while (!this.accessEventQueue.isEmpty()) {
				event = (ResourceAccessEvent) this.accessEventQueue.take();
				logger.debug("recd event with TS {} ", event.getTimestamp());
				// TODO transform the event into a specific log file format

				Thread.sleep(50);
			}
		} catch (InterruptedException e) {
			logger.debug("{}", e.getMessage());
		}
		logger.debug("Shut down");
	}

}
