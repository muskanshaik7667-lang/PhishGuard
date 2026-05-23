from flask import Flask, render_template, request, jsonify
import numpy as np
import re
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Dataset ──────────────────────────────────────────────────────────────────
PHISHING_EMAILS = [
    "URGENT: Your account has been suspended! Click here immediately to verify your identity or lose access forever. http://secure-login-verify.xyz/account",
    "Congratulations! You've won $1,000,000 in our lottery! Send your bank details to claim your prize now! Reply immediately to winner@prize-claim.net",
    "Dear Customer, Your PayPal account needs immediate verification. Visit http://paypal-secure-update.tk/login to avoid suspension",
    "WARNING: Suspicious activity detected on your account. Verify your credentials immediately at http://bank-security-alert.com/verify",
    "Your Amazon order has been compromised. Update payment info at http://amazon-account-secure.xyz/payment",
    "IRS Notice: You owe back taxes. Pay immediately at http://irs-payment-secure.net/pay or face arrest",
    "Your Microsoft account will be closed in 24 hours. Click here to prevent closure: http://microsoft-account-verify.tk",
    "WINNER NOTIFICATION: You have been selected for $500 gift card. Claim now! Limited time offer expires soon. Visit http://gift-card-claim.net",
    "Urgent security alert: Your email password was compromised. Reset immediately http://email-password-reset.xyz/secure",
    "Dear valued customer, verify your credit card information to continue using our services http://creditcard-verify.net/secure",
    "Your bank account shows suspicious transactions. Login immediately to verify: http://chase-bank-secure-login.xyz",
    "Nigerian Prince needs your help to transfer $25 million. You will receive 30% commission. Send bank details to prince@nigeria-transfer.com",
    "Your Apple ID has been locked due to too many failed login attempts. Verify at http://apple-id-unlock.tk/verify",
    "FEDEX: Your package is on hold. Pay customs fee immediately at http://fedex-customs-payment.net",
    "Congratulations! Your Netflix subscription is expiring. Update payment at http://netflix-billing-update.xyz/payment",
    "SECURITY ALERT: Someone tried to access your Gmail. Verify identity immediately: http://gmail-security-verify.tk",
    "You have been selected for a COVID-19 relief fund of $3,500. Claim your money at http://covid-relief-funds.net/claim",
    "Your LinkedIn profile needs immediate verification. Click here: http://linkedin-verify-account.xyz",
    "IMPORTANT: Your domain is about to expire. Renew now at discount prices http://domain-renew-cheap.net/renew",
    "Walmart Gift Card Winner! You've won a $1000 Walmart card. Click to claim: http://walmart-winner.xyz/claim",
    "HSBC Bank Alert: Unauthorized access detected. Verify account at http://hsbc-secure-access.tk/login",
    "Your Dropbox storage is full. Upgrade now for free at http://dropbox-free-upgrade.net",
    "TAX REFUND: You are eligible for a $2,400 tax refund. Claim at http://irs-refund-claim.xyz",
    "DHL DELIVERY: Package held at customs. Pay $2.99 shipping fee: http://dhl-customs-fee.net/pay",
    "Your Venmo account has been compromised. Secure it now: http://venmo-account-secure.xyz/verify",
]

SAFE_EMAILS = [
    "Hi John, I wanted to follow up on our meeting from Tuesday. Please find attached the quarterly report we discussed.",
    "Dear Team, Please join us for the company picnic this Saturday at 2 PM in Central Park. Food and drinks will be provided.",
    "Your Amazon order #123-456-789 has been shipped and will arrive by Thursday. Track your package on our website.",
    "Hi Sarah, Thanks for your application. We'd like to schedule an interview for next week. Please let us know your availability.",
    "Monthly Newsletter: Check out our latest blog posts on productivity tips and upcoming webinar schedule for Q4.",
    "Reminder: Your dentist appointment is scheduled for tomorrow at 3:30 PM. Please call if you need to reschedule.",
    "Thank you for your purchase! Your receipt for $45.99 is attached. Contact support if you have any questions.",
    "Hi Mom, Just wanted to check in and see how you're doing. Let me know if you're free for a call this weekend.",
    "Team update: We've successfully deployed the new features to production. Great work everyone on this sprint!",
    "Your subscription to The New York Times has been renewed for another year. Thank you for your continued support.",
    "Meeting notes from yesterday's standup: Discussed project timeline, assigned tasks to team members, next meeting Friday.",
    "Congratulations on your work anniversary! It's been 3 years since you joined our team. We appreciate your contributions.",
    "Hi there, your library books are due back next Friday. You can renew them online at the library website.",
    "Project status update: Phase 1 complete, Phase 2 starts Monday. Please review the attached timeline document.",
    "Welcome to the book club! Our first meeting is next Thursday at 7 PM. We'll be discussing 'The Midnight Library'.",
    "Your flight AA1234 to New York on Dec 15th is confirmed. Check-in opens 24 hours before departure.",
    "Hi David, here are the minutes from today's board meeting. Please review and let me know if anything needs correction.",
    "Thank you for attending our conference! Slides from all sessions are now available on our website for download.",
    "Friendly reminder: Please submit your expense reports by end of month. Contact HR if you need the submission form.",
    "Your GitHub pull request has been reviewed and approved. The changes have been merged into the main branch.",
    "Good morning! Today's weather forecast: Sunny skies with a high of 72°F. Perfect day for outdoor activities!",
    "Your weekly fitness summary: 15,234 steps, 5 workouts completed, 2,100 calories burned. Great job this week!",
    "Hi Professor Smith, I wanted to ask about the assignment deadline extension. I've been dealing with a family emergency.",
    "The quarterly earnings call will be held on October 15th at 10 AM EST. Dial-in details are attached to this email.",
    "Your Spotify Wrapped is here! You listened to 32,000 minutes of music this year. Your top artist was Taylor Swift.",
]

labels = [1] * len(PHISHING_EMAILS) + [0] * len(SAFE_EMAILS)
emails = PHISHING_EMAILS + SAFE_EMAILS

# ── Feature Engineering ───────────────────────────────────────────────────────
def extract_features(email_list):
    features = []
    for email in email_list:
        text_lower = email.lower()
        feat = {
            'url_count': len(re.findall(r'http[s]?://', email)),
            'suspicious_url': int(bool(re.search(r'http[s]?://[^\s]*\.(xyz|tk|net|info|biz|cc)', email))),
            'urgent_words': sum(1 for w in ['urgent', 'immediate', 'verify', 'suspended', 'warning', 'alert', 'now', 'immediately'] if w in text_lower),
            'money_words': sum(1 for w in ['$', 'prize', 'winner', 'million', 'lottery', 'reward', 'free', 'cash', 'money', 'gift'] if w in text_lower),
            'action_words': sum(1 for w in ['click', 'login', 'verify', 'update', 'confirm', 'access', 'claim'] if w in text_lower),
            'threat_words': sum(1 for w in ['suspended', 'closed', 'blocked', 'arrested', 'compromised', 'unauthorized'] if w in text_lower),
            'exclamation_count': email.count('!'),
            'caps_ratio': sum(1 for c in email if c.isupper()) / max(len(email), 1),
            'has_ip_url': int(bool(re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', email))),
            'long_url': int(bool(re.search(r'http[s]?://\S{30,}', email))),
            'suspicious_tld': int(bool(re.search(r'\.(xyz|tk|net|cc|biz|info|pw|top)/', email))),
            'bank_words': sum(1 for w in ['bank', 'account', 'credit', 'debit', 'paypal', 'payment'] if w in text_lower),
        }
        features.append(list(feat.values()))
    return np.array(features)

# ── Train Model ───────────────────────────────────────────────────────────────
def train_model():
    X_text = emails
    X_manual = extract_features(emails)
    y = np.array(labels)

    X_train_t, X_test_t, X_train_m, X_test_m, y_train, y_test = train_test_split(
        X_text, X_manual, y, test_size=0.25, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train_t)
    X_test_tfidf  = tfidf.transform(X_test_t)

    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_m)])
    X_test_combined  = hstack([X_test_tfidf,  csr_matrix(X_test_m)])

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train_combined, y_train)

    y_pred = clf.predict(X_test_combined)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=['Safe', 'Phishing'], output_dict=True)

    return clf, tfidf, acc, cm, report

clf, tfidf, model_accuracy, conf_matrix, class_report = train_model()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    accuracy_pct = round(model_accuracy * 100, 2)
    cm = conf_matrix
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    metrics = {
        'accuracy': accuracy_pct,
        'precision': round(class_report['Phishing']['precision'] * 100, 2),
        'recall':    round(class_report['Phishing']['recall']    * 100, 2),
        'f1':        round(class_report['Phishing']['f1-score']  * 100, 2),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'total_emails': len(emails),
        'phishing_count': sum(labels),
        'safe_count': len(labels) - sum(labels),
    }
    return render_template('index.html', metrics=metrics)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    email_text = data.get('email', '')
    if not email_text.strip():
        return jsonify({'error': 'No email provided'}), 400

    manual_feat = extract_features([email_text])
    tfidf_feat  = tfidf.transform([email_text])
    combined    = hstack([tfidf_feat, csr_matrix(manual_feat)])

    prediction = clf.predict(combined)[0]
    proba      = clf.predict_proba(combined)[0]
    confidence = round(float(max(proba)) * 100, 1)

    text_lower = email_text.lower()
    indicators = []
    if len(re.findall(r'http[s]?://', email_text)) > 0:
        indicators.append({'type': 'url', 'text': 'Contains URLs'})
    if re.search(r'http[s]?://[^\s]*\.(xyz|tk|net|info|biz|cc)', email_text):
        indicators.append({'type': 'suspicious_url', 'text': 'Suspicious domain extension'})
    urgent = [w for w in ['urgent', 'immediate', 'warning', 'alert'] if w in text_lower]
    if urgent:
        indicators.append({'type': 'urgent', 'text': f'Urgent language: {", ".join(urgent)}'})
    money = [w for w in ['prize', 'winner', 'lottery', 'reward', 'million'] if w in text_lower]
    if money:
        indicators.append({'type': 'money', 'text': f'Money-related words: {", ".join(money)}'})
    if email_text.count('!') >= 2:
        indicators.append({'type': 'exclamation', 'text': f'{email_text.count("!")} exclamation marks'})
    if sum(1 for c in email_text if c.isupper()) / max(len(email_text), 1) > 0.15:
        indicators.append({'type': 'caps', 'text': 'Excessive capitalization'})
    threats = [w for w in ['suspended', 'blocked', 'arrested', 'compromised'] if w in text_lower]
    if threats:
        indicators.append({'type': 'threat', 'text': f'Threat words: {", ".join(threats)}'})

    return jsonify({
        'classification': 'Phishing' if prediction == 1 else 'Safe',
        'confidence': confidence,
        'phishing_prob': round(float(proba[1]) * 100, 1),
        'safe_prob':     round(float(proba[0]) * 100, 1),
        'indicators': indicators,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)