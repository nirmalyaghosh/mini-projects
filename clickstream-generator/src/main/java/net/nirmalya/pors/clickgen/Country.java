package net.nirmalya.pors.clickgen;

/**
 * Represents the country codes. Used to refer to specific configuration files.
 * 
 * @author Nirmalya Ghosh
 */
public enum Country {
	AUSTRALIA("AU"), 
	CHINA("CN"), 
	INDIA("IN"), 
	INDONESIA("ID"), 
	MYANMAR("MM"), 
	NEW_ZEALAND("NZ"), 
	SINGAPORE("SG"), 
	UNITED_STATES("US");

	private String code;

	private Country(String countryCode) {
		this.code = countryCode;
	}

	public String getCode() {
		return code;
	}
}
