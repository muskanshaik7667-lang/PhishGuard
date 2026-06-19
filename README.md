# 🛡️ PhishGuard

**ML-Powered Email Threat Classifier**

PhishGuard is a machine learning-based cybersecurity application that detects and classifies phishing emails and websites in real-time. Using advanced feature engineering and a Random Forest classifier combined with TF-IDF vectorization, PhishGuard analyzes suspicious content patterns including URLs, keywords, and email text characteristics to identify potential cyber threats.

# ✨ Features

- Real-Time Phishing Detection: Classify emails as phishing or safe in milliseconds
- Advanced Feature Analysis: Combines TF-IDF text vectorization with 12+ hand-crafted security features
- Comprehensive Threat Indicators: Identifies suspicious URLs, urgent language, money-related terms, threat words, and more
- Interactive Web Interface: Modern, responsive UI with real-time visualization of predictions
- Model Transparency: View confidence scores, probability distributions, and detailed threat indicators for each classification
- Performance Metrics Dashboard: Monitor model accuracy, precision, recall, F1-score, and confusion matrix in real-time

 # Usage
**Web Interface**
Input Email: Paste the email text you want to analyze into the input textarea
Analyze: Click the "Analyze Email" button
Review Results:
View the classification (Phishing/Safe)
Check the confidence percentage
Review individual threat indicators
View probability distribution
Quick Test Samples
Use the quick test buttons to try pre-loaded samples:

- 🎣 Lottery Scam - Example phishing email
- 🏦 Bank Phish - Banking fraud attempt
- ⚠️ Account Suspend - Account verification scam
- ✅ Work Email - Legitimate business email
- 📦 Order Confirm - Order confirmation email
- 📅 Meeting Invite - Calendar invitation email

🛠️ Technical Stack
Backend: Python, Flask
Machine Learning: scikit-learn, NumPy, SciPy
Frontend: HTML5, CSS3, Vanilla JavaScript
Deployment Ready: Compatible with standard Python hosting platforms
🎯 Use Cases
- Email Security Gateway: Integrate into email systems for automatic phishing detection
- Security Awareness Training: Use as a tool to educate users about phishing threats
- Cybersecurity Research: Analyze phishing patterns and threat evolution
- Security Auditing: Evaluate email security posture of organizations
