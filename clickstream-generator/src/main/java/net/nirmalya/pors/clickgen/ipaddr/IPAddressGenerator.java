package net.nirmalya.pors.clickgen.ipaddr;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import net.nirmalya.pors.clickgen.Country;

import org.apache.commons.net.util.SubnetUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Generates IP addresses based on randomly selected address block(s) read from country specific file(s) in the resource
 * directory.
 * 
 * @author Nirmalya Ghosh
 */
public class IPAddressGenerator {

	private final boolean useSampleFile = true;
	private static final IPAddressGenerator INSTANCE = new IPAddressGenerator();
	private static Logger logger = LoggerFactory.getLogger(IPAddressGenerator.class);
	private Map<Integer, String> lines = new HashMap<Integer, String>();
	private Random random = new Random();
	private String countryCodeLoaded;

	public static String generateIPAddress(Country countryCode) {
		return INSTANCE.generate(countryCode);
	}

	public static String[] generateIPAddresses(Country countryCode, int count) {
		String[] ipAddresses = new String[count];
		for (int i = 0; i < count; i++) {
			ipAddresses[i] = INSTANCE.generate(countryCode);
		}
		return ipAddresses;
	}

	public static void main(String[] args) {
		IPAddressGenerator.generateIPAddress(Country.SINGAPORE);
	}

	private IPAddressGenerator() {
	}

	private String generate(Country countryCode) {
		loadCidrFile(countryCode);
		int lineNumber = 1 + this.random.nextInt(this.lines.size());
		String cidr = this.lines.get(lineNumber).trim();
		while (cidr.endsWith("/31") || cidr.endsWith("/32")) {
			// TODO figure out why exceptions are thrown when /31 and /32
			cidr = this.lines.get(this.random.nextInt(this.lines.size()));
		}

		SubnetUtils util = new SubnetUtils(cidr);
		String[] ipAddresses = util.getInfo().getAllAddresses();
		int index = this.random.nextInt(ipAddresses.length);
		String ipAddress = ipAddresses[index];
		logger.debug("Selected {} (#{} of {} IP addresses based on {})", ipAddress, index, ipAddresses.length, cidr);
		return ipAddress;
	}

	private void loadCidrFile(Country country) {
		if ((this.countryCodeLoaded != null) && (this.countryCodeLoaded.equals(country.getCode()))) {
			return;
		}

		this.lines.clear();
		String cidrFileName = String.format((useSampleFile ? "sample-" : "") + "cidr-%s.txt", country.getCode());
		InputStream inputStream = IPAddressGenerator.class.getResourceAsStream(cidrFileName);
		int lineNumber = 0;
		try (Scanner scanner = new Scanner(inputStream)) {
			while (scanner.hasNextLine()) {
				this.lines.put(++lineNumber, scanner.nextLine());
			}
			this.countryCodeLoaded = country.getCode();
			logger.debug("{} lines read from {}", this.lines.size(), cidrFileName);
		} catch (Exception e) {
			logger.debug("Caught an exception whilst reading {} {}", cidrFileName, e);
		}
	}
}
