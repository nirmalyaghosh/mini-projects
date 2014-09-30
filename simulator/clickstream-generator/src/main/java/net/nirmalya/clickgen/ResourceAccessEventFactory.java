package net.nirmalya.clickgen;

import java.math.BigInteger;
import java.security.SecureRandom;

/**
 * 
 * @author Nirmalya Ghosh
 */
public class ResourceAccessEventFactory {

	private static SecureRandom random = new SecureRandom();

	public static ResourceAccessEvent constructEvent(long timestamp, String remoteHostAddress,
			String resourceRequested, String refererredByResource, String cookieName, String cookieValue) {

		String cookieNameValueString = ((cookieName != null && cookieValue != null) ? String.format("%s=%s",
				cookieName, cookieValue) : null);
		ResourceAccessEvent accessEvent = new ResourceAccessEvent(timestamp, remoteHostAddress, null,
				resourceRequested, Math.max(1250, random.nextInt(15000)), refererredByResource, cookieNameValueString);

		return accessEvent;
	}

	public static String generateCookie() {
		String cookie;
		synchronized (ResourceAccessEventFactory.class) {
			cookie = new BigInteger(130, random).toString(32);
		}

		return cookie;
	}

}
