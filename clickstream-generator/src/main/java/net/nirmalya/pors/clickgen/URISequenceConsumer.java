package net.nirmalya.pors.clickgen;

import java.util.concurrent.BlockingQueue;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import net.nirmalya.pors.clickgen.ipaddr.IPAddressGenerator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Consumes the sequence of URIs produced by the {@link URISequenceProducer}.
 * 
 * @author Nirmalya Ghosh
 */
public class URISequenceConsumer implements Runnable {

	private BlockingQueue<ResourceAccessEvent> accessEventQueue = null;
	private BlockingQueue<String> queue = null;
	private static Logger logger = LoggerFactory.getLogger(URISequenceConsumer.class);

	public URISequenceConsumer(BlockingQueue<String> uriSequenceQueue,
			BlockingQueue<ResourceAccessEvent> accessEventQueue) {
		this.queue = uriSequenceQueue;
		this.accessEventQueue = accessEventQueue;
	}

	@Override
	public void run() {
		logger.debug("Started");

		Random rng = new Random(); // source of randomness
		final String SHUTDOWN_REQ = "Shutdown";
		String item;
		try {
			while ((item = (String) this.queue.take()) != SHUTDOWN_REQ) {
				processURISequence(item, rng);
			}
			logger.debug("Shutting down");
		} catch (InterruptedException e) {
			logger.debug("{}", e.getMessage());
		}

		logger.debug("Shut down");
	}

	private void processURISequence(String uriSequence, Random rng) throws InterruptedException {
		logger.debug("Processing sequence : {}", uriSequence);
		// Separate the sequence into individual URIs
		String[] uris = uriSequence.replace("[", "").replace("]", "").split(", ");

		// Construct events representing access to each of the URIs
		// In addition, this will be done for one or more countries
		List<Country> list = randomlySelectCountries(rng, Math.min(5, uris.length));
		logger.debug("This sequence will apply for simulated users in the following {} countries : {}", list.size(),
				list);

		// Generate IP addresses and cookies for each of the countries
		Map<String, String[]> countryIPAddresses = new HashMap<String, String[]>();
		Map<String, String> ipAddressCookieMap = new HashMap<String, String>();
		Map<String, Long> ipAddressTimestampMap = new HashMap<String, Long>();
		for (Country country : list) {
			int numIPAddresses = Math.max(1, rng.nextInt(list.size()));
			String[] ipAddresses = IPAddressGenerator.generateIPAddresses(country, numIPAddresses);
			logger.trace("{} : {} ipAddresses for {}", country, ipAddresses.length);
			countryIPAddresses.put(country.name(), ipAddresses);
			for (String ipAddress : ipAddresses) {
				ipAddressCookieMap.put(ipAddress, ResourceAccessEventFactory.generateCookie());

				// Initialize the simulated system time for each IP address
				ipAddressTimestampMap.put(ipAddress, System.currentTimeMillis() - Math.max(60000, rng.nextInt(300000)));
			}
		}

		// Generate the events
		boolean isFirstUri = true;
		String previousUri = null;
		for (String uri : uris) {
			for (Country country : list) {
				// Get the IP address and cookie to be used
				String countryName = country.name();
				String[] ipAddressesToBeUsed = countryIPAddresses.get(countryName);

				// Generate the event
				for (String ipAddress : ipAddressesToBeUsed) {
					String cookieToBeUsed = ipAddressCookieMap.get(ipAddress);
					logger.debug("{} {} {} {}", uri, country, ipAddress, cookieToBeUsed);// TODO
					String refererredByResource = null;
					if (!isFirstUri) {
						refererredByResource = previousUri;
					}
					long timestamp = ipAddressTimestampMap.get(ipAddress);
					ResourceAccessEvent accessEvent = ResourceAccessEventFactory.constructEvent(timestamp, ipAddress,
							uri, refererredByResource, "cookieName", cookieToBeUsed);
					this.accessEventQueue.put(accessEvent);

					// Simulated user time (minimum 10 seconds on page)
					timestamp += Math.max(10000, rng.nextInt(300000));

					// Move to the next URI in the sequence
					previousUri = uri;
				}

			}

			if (isFirstUri) {
				isFirstUri = false;
			}
		}

	}

	/**
	 * Randomly selects one or more countries.
	 */
	private List<Country> randomlySelectCountries(Random rng, int numCountriesToSelect) {
		List<Country> cl = Arrays.asList(Country.values());
		List<Country> copy = new LinkedList<Country>(cl);
		Collections.shuffle(copy);
		return copy.subList(0, numCountriesToSelect);
	}
}
