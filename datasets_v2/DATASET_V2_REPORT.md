# Dataset V2 Report

## Train V2
- Rows: `2420`
- Labels: `{'benign': 1210, 'malicious': 1210}`
- Top sources: `[('bipia', 291), ('jackhhao', 272), ('safeguard', 262), ('deepset', 243), ('InjecGuard-NotInject', 220), ('ailuminate', 220), ('harmbench', 220), ('tensor_trust', 150), ('asb', 150), ('cyberseceval2', 130), ('synthetic', 114), ('transfer_attack', 61)]`
- Top categories: `[('benign', 660), ('over_defense', 220), ('benign_table', 89), ('prompt_injection', 86), ('Disruptive Attack', 85), ('prompt_hijacking', 77), ('prompt_extraction', 73), ('Stealthy Attack', 65), ('copyright', 59), ('jailbreak', 52), ('benign_email', 50), ('benign_code', 49)]`

## Eval Hard
- Rows: `728`
- Labels: `{'malicious': 364, 'benign': 364}`
- Top sources: `[('jackhhao', 383), ('safeguard', 172), ('tensor_trust', 72), ('deepset', 45), ('ivanleomk', 36), ('DMPI-PMHFE', 8), ('Design Patterns', 5), ('Protocol Exploits', 3), ('Multi-Agent Defense', 2), ('Tool Result Parsing', 2)]`
- Top categories: `[('benign', 364), ('prompt_injection', 209), ('jailbreak', 83), ('prompt_extraction', 37), ('prompt_hijacking', 35)]`

## Dropped During Train V2 Prep
- `{'placeholder': 41, 'tiny_outlier': 41, 'base64_like': 1}`

## Notes
- Train v2 removes placeholder/template artifacts and tiny outliers.
- Train v2 is balanced by label and capped by source+label to reduce shortcut learning.
- Eval hard emphasizes difficult attack styles and outlier formats.
