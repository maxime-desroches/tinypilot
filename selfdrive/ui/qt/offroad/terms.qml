import QtQuick 2.0

Item {
	id: root
	width: 1920
	height: 840
  signal qmlSignal(variant msg)

	Rectangle {
			color: "black"
			width: parent.width
      height: parent.height
	}

	Flickable {
		id: flickArea
    objectName: "flickArea"
		anchors.fill: parent
		contentHeight: helpText.height
		contentWidth: helpText.width
		flickableDirection: Flickable.VerticalFlick
    flickDeceleration: 7500.0
    maximumFlickVelocity: 10000.0


    onAtYEndChanged: root.qmlSignal("Hello")

		Text {
			// HTML like markup can also be used
			id: helpText
			width: 1800
			font.pointSize: 12
			textFormat: Text.RichText
      color: "white"
			wrapMode: Text.Wrap

			text: '<!DOCTYPE html>

						<head>
							<meta charset="utf-8" />
							<title>openpilot Terms of Service</title>
							<style type="text/css">
								body {
									line-height: 1.7;
								}
							</style>
						</head>

						<body>
							<h2 id="openpilot-terms-and-conditions">openpilot Terms and Conditions</h2>
							<p>The Terms and Conditions below are effective for all users</p>
							<p>Last Updated on October 18, 2019</p>
							<p>Please read these Terms of Use (“Terms”) carefully before using openpilot which is open-sourced software developed by Comma.ai, Inc., a corporation organized under the laws of Delaware (“comma,” “us,” “we,” or “our”).</p>
							<p><strong>Before using and by accessing openpilot, you indicate that you have read, understood, and agree to these Terms.</strong> These Terms apply to all users and others who access or use openpilot. If others use openpilot through your user account or vehicle, you are responsible to ensure that they only use openpilot when it is safe to do so, and in compliance with these Terms and with applicable law. If you disagree with any part of the Terms, you should not access or use openpilot.</p>
							<h3 id="communications">Communications</h3>
							<p>You agree that comma may contact you by email or telephone in connection with openpilot or for other business purposes. You may opt out of receiving email messages at any time by contacting us at support@comma.ai.</p>
							<p>We collect, use, and share information from and about you and your vehicle in connection with openpilot. You consent to comma accessing the systems associated with openpilot, without additional notice or consent, for the purposes of providing openpilot, data collection, software updates, safety and cybersecurity, suspension or removal of your account, and as disclosed in the Privacy Policy (available at https://my.comma.ai/privacy).</p>
							<h3 id="safety">Safety</h3>
							<p>openpilot performs the functions of Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS) designed for use in compatible motor vehicles. While using openpilot, it is your responsibility to obey all laws, traffic rules, and traffic regulations governing your vehicle and its operation. Access to and use of openpilot is at your own risk and responsibility, and openpilot should be accessed and/or used only when you can do so safely.</p>
							<p>openpilot does not make your vehicle “autonomous” or capable of operation without the active monitoring of a licensed driver. It is designed to assist a licensed driver. A licensed driver must pay attention to the road, remain aware of navigation at all times, and be prepared to take immediate action. Failure to do so can cause damage, injury, or death.</p>
							<h3 id="supported-locations-and-models">Supported Locations and Models</h3>
							<p>openpilot is compatible only with particular makes and models of vehicles. For a complete list of currently supported vehicles, visit https://comma.ai. openpilot will not function properly when installed in an incompatible vehicle. openpilot is compatible only within the geographical boundaries of the United States of America.</p>
							<h3 id="indemnification">Indemnification</h3>
							<p><strong>To the maximum extent allowable by law, you agree to defend, indemnify and hold harmless comma, and its employees, partners, suppliers, contractors, investors, agents, officers, directors, and affiliates, from and against any and all claims, damages, causes of action, penalties, interest, demands, obligations, losses, liabilities, costs or debt, additional taxes, and expenses (including but not limited to attorneys’ fees), resulting from or arising out of (i) your use and access of, or inability to use or access, openpilot, (ii) your breach of these Terms, (iii) the inaccuracy of any information, representation or warranty made by you, (iv) activities of anyone other than you in connection with openpilot conducted through your comma device or account, (v) any other of your activities under or in connection with these Terms or openpilot.</strong></p>
							<h3 id="limitation-of-liability">Limitation of Liability</h3>
							<p>In no event shall comma, nor its directors, employees, partners, agents, suppliers, or affiliates, be liable for any indirect, incidental, special, consequential or punitive damages, including without limitation, loss of profits, data, use, goodwill, or other intangible losses, resulting from (i) your access to or use of or inability to access or use of the Software; or (ii) any conduct or content of any third party on the Software whether based on warranty, contract, tort (including negligence) or any other legal theory, whether or not we have been informed of the possibility of such damage, and even if a remedy set forth herein is found to have failed of its essential purpose.</p>
							<h3 id="no-warranty-or-obligations-to-maintain-or-service">No Warranty or Obligations to Maintain or Service</h3>
							<p>comma provides openpilot without representations, conditions, or warranties of any kind. openpilot is provided on an “AS IS” and “AS AVAILABLE” basis, including with all faults and errors as may occur. To the extent permitted by law and unless prohibited by law, comma on behalf of itself and all persons and parties acting by, through, or for comma, explicitly disclaims all warranties or conditions, express, implied, or collateral, including any implied warranties of merchantability, satisfactory quality, and fitness for a particular purpose in respect of openpilot.</p>
							<p>To the extent permitted by law, comma does not warrant the operation, performance, or availability of openpilot under all conditions. comma is not responsible for any failures caused by server errors, misdirected or redirected transmissions, failed internet connections, interruptions or failures in the transmission of data, any computer virus, or any acts or omissions of third parties that damage the network or impair wireless service.</p>
							<p>We undertake reasonable measures to preserve and secure information collected through our openpilot. However, no data collection, transmission or storage system is 100% secure, and there is always a risk that your information may be intercepted without our consent. <em>In using openpilot, you acknowledge that comma is not responsible for intercepted information, and you hereby release us from any and all claims arising out of or related to the use of intercepted information in any unauthorized manner.</em></p>
							<p>By providing openpilot, comma does not transfer or license its intellectual property or grant rights in its brand names, nor does comma make representations with respect to third-party intellectual property rights.</p>
							<p>We are not obligated to provide any maintenance or support for openpilot, technical or otherwise. If we voluntarily provide any maintenance or support for openpilot, we may stop any such maintenance, support, or services at any time in our sole discretion.</p>
							<h3 id="modification-of-software">Modification of Software</h3>
							<p>In no event shall comma, nor its directors, employees, partners, agents, suppliers, or affiliates, be liable if you choose to modify the software.</p>
							<h3 id="changes">Changes</h3>
							<p>We reserve the right, at our sole discretion, to modify or replace these Terms at any time. If a revision is material we will provide at least 15 days’ notice prior to any new terms taking effect. What constitutes a material change will be determined at our sole discretion.</p>
							<p>By continuing to access or use our Software after any revisions become effective, you agree to be bound by the revised terms. If you do not agree to the new terms, you are no longer authorized to use the Software.</p>
							<h3 id="contact-us">Contact Us</h3>
							<p>If you have any questions about these Terms, please contact us at support@comma.ai.</p>
						</body>
						</html>'
		}
	}
}


