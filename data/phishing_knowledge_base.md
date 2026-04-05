# Phishing Email Analysis Knowledge Base

This knowledge base contains labeled phishing and legitimate email samples for threat identification training. Each entry includes key indicators to support email threat classification.

---

## Phishing Emails

### PHISH-001: PayPal Account Compromise Scam
- **Sender:** security@paypa1-support.com
- **Subject:** Urgent: Your Account Has Been Compromised
- **Body:** Dear Customer, We have detected unauthorized access to your PayPal account. Your account will be permanently suspended within 24 hours unless you verify your identity immediately. Click the link below to secure your account now.
- **URLs:** http://paypa1-support.com/verify-account
- **Has Attachment:** No
- **Urgency Level:** High
- **Attack Type:** Credential Harvesting
- **Key Indicators:** Spoofed domain (paypa1 vs paypal); high urgency with 24-hour deadline; generic greeting; suspicious URL not matching official PayPal domain

### PHISH-002: Microsoft Password Expiration Scam
- **Sender:** admin@micr0soft-security.com
- **Subject:** Action Required: Password Expiration Notice
- **Body:** Your Microsoft 365 password will expire today. To avoid losing access to your email and files, please update your password immediately by clicking the link below. Failure to act will result in account lockout.
- **URLs:** http://micr0soft-security.com/password-reset
- **Has Attachment:** No
- **Urgency Level:** High
- **Attack Type:** Credential Harvesting
- **Key Indicators:** Spoofed domain using zero instead of 'o'; password expiration scare tactic; threatening account lockout; non-official Microsoft URL

### PHISH-003: Business Email Compromise (CEO Fraud)
- **Sender:** ceo@company-mail.net
- **Subject:** Confidential - Wire Transfer Needed ASAP
- **Body:** Hi, I need you to process an urgent wire transfer of $45,000 to a new vendor. I'm currently in a meeting and can't talk, but this needs to be done before end of day. I'll send the details separately. Please confirm you can handle this. - CEO
- **URLs:** None
- **Has Attachment:** No
- **Urgency Level:** High
- **Attack Type:** BEC (Business Email Compromise)
- **Key Indicators:** CEO impersonation; unusual request channel; urgency pressure; request to bypass normal approval process; vague details with promise of follow-up

### PHISH-004: Fake Amazon Order Confirmation
- **Sender:** billing@amaz0n-orders.com
- **Subject:** Your Order #392-4851290 Has Been Placed - $1,299.99
- **Body:** Thank you for your order of MacBook Pro 16-inch ($1,299.99). If you did not make this purchase, click here immediately to cancel and request a refund. Your credit card will be charged within 2 hours.
- **URLs:** http://amaz0n-orders.com/cancel-order?id=392
- **Has Attachment:** No
- **Urgency Level:** High
- **Attack Type:** Credential Harvesting
- **Key Indicators:** Spoofed Amazon domain with zero; fake high-value order to create panic; 2-hour urgency; link to fake cancellation page designed to harvest credentials

### PHISH-005: Fake HR Benefits Update
- **Sender:** hr@company-benefits-portal.com
- **Subject:** Employee Benefits Update - Immediate Action Required
- **Body:** Dear Employee, Our benefits provider has updated their system. To continue receiving your health insurance coverage, please log in to the new portal and re-enter your personal information including SSN and banking details by Friday.
- **URLs:** http://company-benefits-portal.com/update
- **Has Attachment:** No
- **Urgency Level:** Medium
- **Attack Type:** Credential Harvesting
- **Key Indicators:** External domain posing as internal HR; requesting SSN and banking details via email; deadline pressure; no mention of specific company name

### PHISH-006: Fake FedEx Delivery Notification
- **Sender:** delivery@fedx-tracking.com
- **Subject:** FedEx: Delivery Failed - Package Returned
- **Body:** We attempted to deliver your package but no one was available. Please review the attached shipping label and reschedule delivery. If not claimed within 48 hours, the package will be returned to sender.
- **URLs:** http://fedx-tracking.com/reschedule
- **Has Attachment:** Yes (malicious file disguised as shipping label)
- **Urgency Level:** Medium
- **Attack Type:** Malware Delivery
- **Key Indicators:** Misspelled brand (fedx vs fedex); malicious attachment disguised as shipping label; 48-hour pressure; no tracking number provided

### PHISH-007: Fake Invoice Scam
- **Sender:** invoice@quickbooks-invoicing.net
- **Subject:** Invoice #INV-7823 Due - Payment Overdue
- **Body:** Please find attached the overdue invoice #INV-7823 for $3,450.00. Payment was due on March 15. To avoid late fees and service interruption, please process payment immediately or download the attached invoice for your records.
- **URLs:** None
- **Has Attachment:** Yes (malicious invoice attachment)
- **Urgency Level:** Medium
- **Attack Type:** Invoice Fraud
- **Key Indicators:** Fake QuickBooks domain; unsolicited invoice with no prior business relationship; malicious attachment; threatening late fees

### PHISH-008: Fake Dropbox File Sharing
- **Sender:** support@dropbox-sharing.com
- **Subject:** John shared a document with you
- **Body:** John Smith has shared a file with you: 'Q4_Financial_Report.pdf'. Click below to view the shared document in your Dropbox account. This link will expire in 24 hours.
- **URLs:** http://dropbox-sharing.com/view/doc-q4report
- **Has Attachment:** No
- **Urgency Level:** Low
- **Attack Type:** Credential Harvesting
- **Key Indicators:** Fake Dropbox sharing notification; non-official domain; link expiration creates subtle urgency; leads to fake login page

### PHISH-009: IRS Tax Refund Scam
- **Sender:** tax-refund@irs-gov-refund.com
- **Subject:** IRS Tax Refund Notification - Refund of $4,872.50
- **Body:** After reviewing your tax filing, you are eligible for a tax refund of $4,872.50. To receive your refund, please verify your identity and banking information through our secure portal within 5 business days.
- **URLs:** http://irs-gov-refund.com/claim-refund
- **Has Attachment:** No
- **Urgency Level:** Medium
- **Attack Type:** Credential Harvesting
- **Key Indicators:** IRS does not send refund notifications by email; fake government domain; requesting banking information; specific dollar amount to seem legitimate

### PHISH-010: Fake VPN Certificate Update
- **Sender:** it-helpdesk@corp-vpn-access.com
- **Subject:** VPN Certificate Renewal Required
- **Body:** Your corporate VPN certificate expires tomorrow. To maintain remote access, download and install the updated certificate from the link below. Contact IT support if you experience any issues.
- **URLs:** http://corp-vpn-access.com/cert-update.exe
- **Has Attachment:** No
- **Urgency Level:** Medium
- **Attack Type:** Malware Delivery
- **Key Indicators:** External domain posing as IT helpdesk; download link points to executable file (.exe); certificate renewal pretext; no internal ticket reference

---

## Legitimate Emails

### LEGIT-001: GitHub Sign-in Notification
- **Sender:** noreply@github.com
- **Subject:** [GitHub] A new sign-in from Chrome on macOS
- **Body:** Hey josephwu, We noticed a new sign-in to your account from Chrome on macOS. If this was you, no further action is needed. If you did not sign in recently, please review your account security settings.
- **URLs:** https://github.com/settings/security
- **Has Attachment:** No
- **Urgency Level:** Low
- **Attack Type:** None
- **Key Indicators:** Official github.com domain; personalized with username; no urgent action demanded; link points to official GitHub settings; calm tone

### LEGIT-002: Google Security Alert
- **Sender:** no-reply@accounts.google.com
- **Subject:** Security alert: New device sign-in
- **Body:** Your Google Account was just signed in to from a new Windows device. If this was you, you don't need to do anything. If not, we'll help you secure your account. You can review your recent activity at myaccount.google.com.
- **URLs:** https://myaccount.google.com/notifications
- **Has Attachment:** No
- **Urgency Level:** Low
- **Attack Type:** None
- **Key Indicators:** Official Google domain; balanced tone without panic; provides context; links to official Google domain; does not request credentials directly

### LEGIT-003: SANS Newsletter
- **Sender:** newsletter@sans.org
- **Subject:** SANS NewsBites - Weekly Cybersecurity News Summary
- **Body:** This week's top stories: Critical vulnerability in Apache Log4j patched; NIST releases updated cybersecurity framework; Ransomware attacks targeting healthcare sector increase 40%. Read the full newsletter at sans.org.
- **URLs:** https://www.sans.org/newsletters/newsbites
- **Has Attachment:** No
- **Urgency Level:** Low
- **Attack Type:** None
- **Key Indicators:** Known cybersecurity organization; informational content; no action required; official domain; consistent with expected newsletter format

### LEGIT-004: AWS Monthly Bill
- **Sender:** aws-notifications@amazon.com
- **Subject:** Your AWS Monthly Bill - March 2026
- **Body:** Your AWS bill for March 2026 is now available. Total charges: $127.43. You can view your detailed billing statement in the AWS Billing Console. Payment will be processed automatically using your default payment method.
- **URLs:** https://console.aws.amazon.com/billing
- **Has Attachment:** Yes (PDF billing statement)
- **Urgency Level:** Low
- **Attack Type:** None
- **Key Indicators:** Official Amazon domain; expected monthly billing email; links to official AWS console; attachment is PDF bill; no urgent action required

### LEGIT-005: Slack Weekly Digest
- **Sender:** team@slack.com
- **Subject:** Weekly digest for NTHU Security Lab workspace
- **Body:** Here's what happened in your Slack workspace this week: 23 messages in #general, 15 messages in #research, 8 messages in #paper-discussion. Top channels and conversations are summarized below.
- **URLs:** https://nthu-security.slack.com
- **Has Attachment:** No
- **Urgency Level:** Low
- **Attack Type:** None
- **Key Indicators:** Official Slack domain; routine weekly summary; workspace-specific content; no action required; informational only

---

## Common Phishing Indicators Summary

| Indicator Category | Phishing Signs | Legitimate Signs |
|---|---|---|
| **Domain** | Misspelled (paypa1, micr0soft, fedx) | Official domain (github.com, google.com) |
| **Urgency** | "Immediately", "within 24 hours", "ASAP" | "No further action needed", "if this was you" |
| **Greeting** | Generic ("Dear Customer", "Dear Employee") | Personalized (username, workspace name) |
| **Request** | Credentials, SSN, banking info, wire transfer | Informational, no sensitive data requested |
| **Tone** | Threatening (suspension, lockout, late fees) | Calm, balanced, reassuring |
| **Links** | HTTP, non-official domains, .exe files | HTTPS, official domains |
| **Attachments** | Unsolicited (invoices, shipping labels) | Expected (monthly bills, reports) |
