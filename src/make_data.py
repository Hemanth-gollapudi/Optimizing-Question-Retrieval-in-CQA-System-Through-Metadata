import os
import re
import argparse
import numpy as np
import pandas as pd

def clean_text(x: str) -> str:
    x = "" if x is None else str(x)
    return re.sub(r"\s+", " ", x).strip()

def generate_synthetic_cqa(n_per_cat: int = 220, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # More intents added for more data + more variation
    categories = {
        "Technology": {
            "intents": [
                {
                    "topic": "WiFi disconnecting",
                    "problem": "laptop keeps dropping WiFi every few minutes",
                    "answers": [
                        "Update Wi-Fi drivers and restart router/modem.",
                        "Forget the Wi-Fi network, reconnect, and reset network settings.",
                        "Disable power saving for the Wi-Fi adapter and test another network.",
                        "Switch to 5GHz band and change router channel to reduce interference.",
                        "Flush DNS and set DNS to public servers (8.8.8.8 / 1.1.1.1).",
                        "Update router firmware and check for ISP outages."
                    ],
                    "keywords": ["wifi", "wireless", "network", "router", "driver", "dns", "channel", "5ghz", "firmware"]
                },
                {
                    "topic": "Phone overheating",
                    "problem": "phone gets hot while charging",
                    "answers": [
                        "Use original charger and avoid using phone while charging.",
                        "Close background apps and reduce screen brightness.",
                        "Remove thick phone case and charge in a cool environment.",
                        "Disable fast charging if it increases heat.",
                        "If overheating is extreme, stop charging and check battery health."
                    ],
                    "keywords": ["overheating", "charger", "battery", "fast charging", "brightness", "apps", "temperature"]
                },
                {
                    "topic": "App crashing",
                    "problem": "app closes suddenly when I open it",
                    "answers": [
                        "Clear app cache/data and restart the device.",
                        "Update the app and OS to latest version.",
                        "Uninstall and reinstall the app; ensure enough storage.",
                        "Check permissions and disable battery optimization for the app.",
                        "If issue started after update, reinstall a stable version."
                    ],
                    "keywords": ["crash", "cache", "reinstall", "update", "storage", "permissions", "bug"]
                },
                {
                    "topic": "Storage full",
                    "problem": "storage is full even after deleting files",
                    "answers": [
                        "Clear temporary files and app caches; empty recycle bin/trash.",
                        "Delete duplicate large files using a storage analyzer tool.",
                        "Move photos/videos to cloud or external drive.",
                        "Remove unused apps and clear WhatsApp/Telegram media.",
                        "Delete offline downloads and old backups."
                    ],
                    "keywords": ["storage", "cache", "temporary", "duplicate", "cloud", "backup", "offline", "analyzer"]
                },
                {
                    "topic": "Blue screen",
                    "problem": "PC shows blue screen after update",
                    "answers": [
                        "Boot into safe mode and uninstall the latest update.",
                        "Run memory diagnostics and disk check (chkdsk).",
                        "Update drivers (GPU/chipset) and remove recent software.",
                        "Use System Restore to roll back to a stable point.",
                        "Note the stop code and troubleshoot the specific driver/hardware."
                    ],
                    "keywords": ["bsod", "safe mode", "update", "driver", "chkdsk", "restore", "stop code"]
                },
                # ✅ Added more intents
                {
                    "topic": "Laptop slow",
                    "problem": "laptop has become very slow and takes long to boot",
                    "answers": [
                        "Disable unnecessary startup apps and check task manager usage.",
                        "Free disk space and run disk cleanup.",
                        "Scan for malware and unwanted software.",
                        "Upgrade to SSD or add RAM if hardware is low.",
                        "Update OS and drivers; remove bloatware."
                    ],
                    "keywords": ["slow", "boot", "startup", "ssd", "ram", "malware", "disk cleanup", "performance"]
                },
                {
                    "topic": "Bluetooth not working",
                    "problem": "Bluetooth is not detecting devices",
                    "answers": [
                        "Turn Bluetooth off/on and restart the device.",
                        "Remove the device and pair again.",
                        "Update Bluetooth drivers and OS.",
                        "Check airplane mode and Bluetooth services.",
                        "Try pairing with another device to isolate issue."
                    ],
                    "keywords": ["bluetooth", "pairing", "drivers", "device not found", "services", "airplane mode"]
                },
            ],
            "subj": [
                "How to fix {topic}?",
                "Need help with {topic}",
                "{topic} on my device",
                "Trouble with {topic}",
                "What should I do about {topic}?"
            ],
            "body": [
                "I am facing {problem}. What can I do to solve it permanently?",
                "This started after a recent change. Details: {problem}. Any suggestions?",
                "Please explain possible causes for {problem} and how to troubleshoot.",
                "I tried basic steps but it didn't work. Issue: {problem}.",
                "It happens mostly in the evening. Problem: {problem}."
            ],
        },

        "Finance": {
            "intents": [
                {
                    "topic": "UPI failed",
                    "problem": "UPI transaction failed but money deducted",
                    "answers": [
                        "Wait 2–24 hours; many failed UPI debits auto-reverse.",
                        "Check status in app; if failed with debit, raise dispute.",
                        "Contact bank with UTR/reference number for reversal.",
                        "If not reversed in 24 hours, file complaint via bank/NPCI.",
                        "Check daily limits, network issues, and app updates."
                    ],
                    "keywords": ["upi", "reversal", "utr", "reference", "bank", "npci", "pending", "limit", "dispute"]
                },
                {
                    "topic": "Budget planning",
                    "problem": "want to track spending and save more",
                    "answers": [
                        "Use 50/30/20 rule and track weekly expenses.",
                        "Automate savings first, then budget the rest.",
                        "Set category limits and review monthly.",
                        "Cut frequent small expenses and set a goal.",
                        "Make separate accounts for bills and savings."
                    ],
                    "keywords": ["budget", "expenses", "savings", "tracking", "50/30/20", "goal", "limits"]
                },
                {
                    "topic": "Credit score",
                    "problem": "credit score dropped suddenly",
                    "answers": [
                        "Check credit report for late payment or new inquiry.",
                        "Keep utilization below 30% and pay dues on time.",
                        "Dispute wrong entries with the credit bureau.",
                        "Avoid applying multiple loans/cards together.",
                        "Maintain long credit history; don’t close old cards."
                    ],
                    "keywords": ["credit score", "report", "utilization", "bureau", "dispute", "inquiry", "dues"]
                },
                {
                    "topic": "Investing",
                    "problem": "want to start investing with small amount",
                    "answers": [
                        "Start SIP in index funds; invest monthly.",
                        "Build emergency fund first, then invest long term.",
                        "Diversify and avoid risky tips.",
                        "Use low-cost funds and stay consistent.",
                        "Learn basics: risk, return, compounding."
                    ],
                    "keywords": ["sip", "index fund", "etf", "diversify", "compounding", "risk", "long term"]
                },
                {
                    "topic": "Loan EMI",
                    "problem": "confused between shorter vs longer EMI",
                    "answers": [
                        "Shorter tenure reduces total interest but EMI higher.",
                        "Longer tenure lowers EMI but increases interest paid.",
                        "Use EMI calculator to compare total payout.",
                        "Pick affordable EMI and prepay when possible.",
                        "Stable income → shorter tenure usually better."
                    ],
                    "keywords": ["emi", "tenure", "interest", "prepay", "calculator", "payout", "loan"]
                },
                # ✅ Added more intents
                {
                    "topic": "Salary delay",
                    "problem": "salary is delayed and employer not responding",
                    "answers": [
                        "Follow up via email with HR and keep written records.",
                        "Check payroll date policy and ask payroll team.",
                        "Escalate to manager/finance if no response.",
                        "If repeated issues, seek legal/official labor advice.",
                        "Plan emergency fund to handle short delays."
                    ],
                    "keywords": ["salary", "delay", "hr", "payroll", "escalation", "policy", "records"]
                },
            ],
            "subj": [
                "How to manage {topic}?",
                "Need advice on {topic}",
                "{topic} issue",
                "Beginner question: {topic}",
                "What should I do if {topic} happens?"
            ],
            "body": [
                "Here is the problem: {problem}. What is the safest approach to handle it?",
                "I need a step-by-step strategy for {problem}.",
                "This started suddenly: {problem}. What should I do first?",
                "Any long-term fix would be helpful. Problem: {problem}.",
                "It happens mostly in the evening. Issue: {problem}."
            ],
        },

        "Health": {
            "intents": [
                {
                    "topic": "Morning headache",
                    "problem": "wake up with headache and dry mouth",
                    "answers": [
                        "Hydrate well and improve sleep routine.",
                        "Reduce stress and check dehydration; see doctor if frequent.",
                        "Avoid screens before bed and keep regular sleep time.",
                        "If snoring/apnea symptoms, consult doctor.",
                        "Track triggers like sleep, diet, caffeine."
                    ],
                    "keywords": ["hydration", "sleep", "stress", "caffeine", "snoring", "apnea", "routine"]
                },
                {
                    "topic": "Stomach pain",
                    "problem": "pain and bloating after meals",
                    "answers": [
                        "Avoid trigger foods and eat smaller meals.",
                        "Try probiotics and reduce fizzy drinks.",
                        "Maintain a food diary to find triggers.",
                        "If severe symptoms, seek medical care.",
                        "Walk lightly after meals; don’t lie down."
                    ],
                    "keywords": ["bloating", "trigger foods", "food diary", "probiotics", "hydration", "gas"]
                },
                # ✅ Added more intents
                {
                    "topic": "Cold and cough",
                    "problem": "sore throat and cough for several days",
                    "answers": [
                        "Drink warm fluids and rest; use steam inhalation.",
                        "Gargle with warm salt water and avoid cold drinks.",
                        "If fever persists or breathing issues, consult a doctor.",
                        "Use humidifier and avoid smoke/dust exposure.",
                        "Maintain hydration and light nutrition."
                    ],
                    "keywords": ["cough", "sore throat", "steam", "hydration", "fever", "salt water", "humidifier"]
                },
            ],
            "subj": [
                "What can cause {topic}?",
                "Advice needed for {topic}",
                "{topic} — what should I do?",
                "Is {topic} serious?"
            ],
            "body": [
                "I have {problem}. What are common reasons and safe home-care tips?",
                "Symptoms: {problem}. When should I see a doctor?",
                "Looking for general guidance for {problem}.",
                "It started recently: {problem}. Any simple remedies?"
            ],
        },

        "Education": {
            "intents": [
                {
                    "topic": "Exam anxiety",
                    "problem": "nervous during exams and forget answers",
                    "answers": [
                        "Practice mock tests and breathing exercises.",
                        "Revise with active recall and sleep well.",
                        "Use past papers and time management.",
                        "Break syllabus into small parts; repeat often.",
                        "If extreme anxiety, talk to mentor/counsellor."
                    ],
                    "keywords": ["mock tests", "breathing", "active recall", "past papers", "time management", "mentor"]
                },
                {
                    "topic": "Improve concentration",
                    "problem": "get distracted while studying",
                    "answers": [
                        "Use Pomodoro (25/5) and keep phone away.",
                        "Use spaced repetition and small daily targets.",
                        "Study in low-distraction environment.",
                        "Plan timetable and track progress daily.",
                        "Sleep well and take breaks."
                    ],
                    "keywords": ["pomodoro", "spaced repetition", "targets", "distractions", "focus", "timetable"]
                },
                # ✅ Added more intents
                {
                    "topic": "Programming basics",
                    "problem": "confused about loops and functions in programming",
                    "answers": [
                        "Practice small problems daily and read examples.",
                        "Learn one concept at a time and write mini programs.",
                        "Use debugging/print statements to understand flow.",
                        "Solve beginner coding questions regularly.",
                        "Watch tutorials and implement what you learn."
                    ],
                    "keywords": ["loops", "functions", "debugging", "practice", "mini programs", "coding"]
                },
            ],
            "subj": [
                "How to handle {topic}?",
                "Tips for {topic}",
                "Help regarding {topic}",
                "{topic}: need guidance"
            ],
            "body": [
                "My situation: {problem}. Please share practical steps and resources.",
                "I am struggling because {problem}. What method works best?",
                "Please recommend a plan for {problem}.",
                "Any simple routine for {problem} that works daily?"
            ],
        },
    }

    # Lexical gap paraphrases
    paraphrases = {
        "wifi": ["wireless", "internet", "network"],
        "disconnecting": ["dropping", "cutting off", "losing connection"],
        "transaction": ["payment", "transfer"],
        "deducted": ["debited", "cut"],
        "anxiety": ["nervousness", "stress", "panic"],
        "budget": ["spending plan", "expense tracking", "money plan"],
        "investing": ["wealth building", "market investing", "putting money in funds"],
        "cough": ["cold cough", "throat cough"],
        "slow": ["laggy", "very slow", "sluggish"],
    }

    extras = [
        "I tried basic steps but it didn't work.",
        "It happens mostly in the evening.",
        "This started after an update.",
        "I need a quick solution because it's urgent.",
        "Any long-term fix would be helpful.",
        "Can you explain in simple steps?",
        "I’m not sure what to do next."
    ]

    def add_lexical_gap(text: str) -> str:
        out = text
        for k, reps in paraphrases.items():
            if re.search(rf"\b{k}\b", out, flags=re.IGNORECASE) and rng.random() < 0.6:
                out = re.sub(rf"\b{k}\b", rng.choice(reps), out, flags=re.IGNORECASE)
        if rng.random() < 0.65:
            out = out + " " + rng.choice(extras)
        return out

    rows = []
    qid = 1

    for cat, meta in categories.items():
        intents = meta["intents"]

        for _ in range(n_per_cat):
            intent = intents[rng.integers(0, len(intents))]
            topic = intent["topic"]
            problem = intent["problem"]
            answer_variant = rng.choice(intent["answers"])

            subject = rng.choice(meta["subj"]).format(topic=topic)
            body = rng.choice(meta["body"]).format(problem=problem)

            subject = add_lexical_gap(subject)
            body = add_lexical_gap(body)

            answer = "Suggested steps: " + answer_variant
            if rng.random() < 0.25:
                answer += " If it continues, consider expert help."

            # ✅ Metadata column: keep keywords separate (comma-separated)
            keywords = ", ".join(intent.get("keywords", []))

            rows.append(
                {
                    "qid": str(qid),
                    "subject": clean_text(subject),
                    "body": clean_text(body),
                    "category": cat,
                    "answer": clean_text(answer),
                    "topic": clean_text(topic),
                    "keywords": clean_text(keywords),  # ✅ metadata added
                }
            )
            qid += 1

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="../data/cqa.csv")
    ap.add_argument("--n_per_cat", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate_synthetic_cqa(n_per_cat=args.n_per_cat, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"[OK] Saved: {args.out} (rows={len(df)})")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
