package net.nirmalya.clickgen;

import java.util.Map;

/**
 * Represents an event created each time a simulated user accesses a web
 * resource.
 * 
 * @author Nirmalya Ghosh
 */
public class ResourceAccessEvent implements Comparable<ResourceAccessEvent> {

	private int bytesSent;
	private long timestamp;
	private String cookiesString;
	private String authenticatedUser;
	private String httpReferer;
	private String resourceRequested;
	private String remoteHost;
	private String userAgentString;

	public ResourceAccessEvent(long timestamp, String remoteHostAddress,
			String authenticatedUser, String resourceRequested, int bytesSent,
			String refererredByResource, Map<String, String> cookies) {
		this.authenticatedUser = (authenticatedUser != null) ? authenticatedUser
				: "-";
		this.bytesSent = bytesSent;
		setCookiesString(cookies);
		this.httpReferer = (refererredByResource != null) ? refererredByResource
				: "-";
		this.remoteHost = remoteHostAddress;
		this.resourceRequested = resourceRequested;
		this.timestamp = timestamp;
	}

	public ResourceAccessEvent(long timestamp, String remoteHostAddress,
			String authenticatedUser, String resourceRequested, int bytesSent,
			String refererredByResource, String cookieNameValueString) {
		this.authenticatedUser = (authenticatedUser != null) ? authenticatedUser
				: "-";
		this.bytesSent = bytesSent;
		this.cookiesString = cookieNameValueString;
		this.httpReferer = (refererredByResource != null) ? refererredByResource
				: "-";
		this.remoteHost = remoteHostAddress;
		this.resourceRequested = resourceRequested;
		this.timestamp = timestamp;
	}

	@Override
	public int compareTo(ResourceAccessEvent other) {
		if (other == null) {
			return 1;
		} else
			return other.getTimestamp() > this.timestamp ? -1 : ((other
					.getTimestamp() == this.timestamp) ? 0 : 1);
	}

	/**
	 * Returns "-" if not set.
	 */
	public String getAuthenticatedUser() {
		return authenticatedUser;
	}

	public int getBytesSent() {
		return bytesSent;
	}

	/**
	 * Returns a semicolon separated string of cookies similar to
	 * <code>USERID=CustomerA;IMPID=01234</code>. Returns "-" if no cookies have
	 * been set.
	 */
	public String getCookiesString() {
		return cookiesString;
	}

	/**
	 * Returns the resource that linked to the resource being requested. Returns
	 * "-" if not set.
	 */
	public String getHttpReferer() {
		return httpReferer;
	}

	public String getRemoteHost() {
		return remoteHost;
	}

	public String getResourceRequested() {
		return resourceRequested;
	}

	/**
	 * Returns the timestamp of the end-time of the request.
	 */
	public long getTimestamp() {
		return timestamp;
	}

	public String getUserAgentString() {
		return userAgentString;
	}

	private void setCookiesString(Map<String, String> cookies) {
		if (cookies == null || cookies.isEmpty()) {
			this.cookiesString = "-";
		} else {
			StringBuilder sb = new StringBuilder();
			boolean isFirst = true;
			for (String cookie : cookies.keySet()) {
				if (isFirst) {
					isFirst = false;
				} else {
					sb.append(";");
				}
				sb.append(cookie);
			}
			this.cookiesString = sb.toString();
		}
	}

}
