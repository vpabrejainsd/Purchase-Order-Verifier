# Purchase-Order-Verifier

A tool I created during my SU'25 internship at IDC Digital. It compares scanned image PDF's of Purchase Orders vs. Native PDF's of those Purchase Orders, highlighting potential mismatches between the items.

# Requirements

- You need a GCP account for the DocumentAI API to work ; I loaded mine with free credits worth $300 for the purposes of the project. Once created, you'll need to create a **private key.json** file that you import and feed into your .env file with the filepath: "export GOOGLE_APPLICATION_CREDENTIALS="./key.json"".

# Running The Project

- Upload the scanned image PDF to the test folder (as shown, the naming convention I followed was "test/test_*.pdf")
- Upload the native PDF to the true folder (as shown, the naming convention I followed was "true/true_*.pdf")
- Navigate to the project directory and run app.py, and you'll be good to go!