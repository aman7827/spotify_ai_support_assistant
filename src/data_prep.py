"""
src/data_prep.py
----------------
Data Preparation Pipeline for SpotifyCares Customer Support AI Assistant.
Extracts customer-message -> support-reply pairs, applies minimal explainable cleaning,
and outputs clean CSV pairs ready for indexing and intent classification.
"""

import os
import re
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spotify_raw_pairs.csv")
CLEAN_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spotify_cleaned_pairs.csv")

def clean_text(text: str) -> str:
    """
    Minimal, explainable cleaning for customer support tweets:
    1. Remove URLs (http/https).
    2. Remove Twitter agent sign-offs (e.g. ^NK, ^JM, ^BP, ^CA).
    3. Remove redundant @SpotifyCares handles at the start.
    4. Normalize multiple whitespace and strip leading/trailing spaces.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\^[A-Z]{2,3}\b', '', text)
    text = re.sub(r'^@SpotifyCares\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_spotify_dataset():
    """
    Builds a robust, realistic dataset of SpotifyCares Q&A conversation pairs
    spanning all 8 target business intents.
    """
    raw_conversations = [
        # Login & Auth Issue
        ("Login & Auth Issue",
         "@SpotifyCares I can't log into my account since morning! It keeps saying invalid password even after resetting it.",
         "Hey there! Sorry to hear that. Can you try clearing your app cache or logging in via an incognito browser window? Let us know if that works! ^NK"),
        ("Login & Auth Issue",
         "@SpotifyCares I'm trying to log in on my new iPhone but not getting the 2FA verification code on SMS.",
         "Hi! Make sure your network connection is stable. If you still don't receive the SMS code, Send us a DM with your account email address and we'll check. ^JM"),
        ("Login & Auth Issue",
         "@SpotifyCares Why does Spotify keep logging me out every time I close the app on Windows 11?",
         "Hey! That sounds frustrating. Try doing a clean reinstallation of the app. Let us know how it goes! ^BP"),
        ("Login & Auth Issue",
         "@SpotifyCares Forgot my password and the reset email isn't showing up in my inbox or spam folder.",
         "Hi there! Check if your account was registered with Facebook or Apple ID instead. Send us a DM if you still need a hand! ^CA"),
        ("Login & Auth Issue",
         "@SpotifyCares Getting error code 408 when trying to sign in on Spotify Web Player.",
         "Hey! Try disabling your browser extensions or VPN temporarily. If the issue persists, let us know your browser version. ^NK"),

        # Subscription & Premium
        ("Subscription & Premium",
         "@SpotifyCares I renewed my Premium subscription yesterday but my account still shows Free status!",
         "Hey! Try logging out and logging back in—that usually syncs your payment status right away. DM us your receipt if it stays Free! ^JM"),
        ("Subscription & Premium",
         "@SpotifyCares How do I add a new member to my Spotify Family Plan?",
         "Hi! The plan manager can send an invite link from their account page under Manage Family Plan. Everyone must reside at the same address. ^BP"),
        ("Subscription & Premium",
         "@SpotifyCares My student discount expired, how do I re-verify my SheerID status?",
         "Hey! You can re-verify your student status at spotify.com/student. Just follow the SheerID verification prompts there. ^CA"),
        ("Subscription & Premium",
         "@SpotifyCares Does Spotify Premium Duo allow two separate accounts with independent playlists?",
         "Hi there! Yes, Premium Duo provides two distinct Premium accounts under one discounted bill! ^NK"),
        ("Subscription & Premium",
         "@SpotifyCares I paid for 3 months prepaid card but Premium is only active for 1 month.",
         "Hey! Send us a DM with a photo of your gift card receipt and PIN, and we'll investigate for you. ^JM"),

        # Billing & Refund Request
        ("Billing & Refund Request",
         "@SpotifyCares You guys charged me twice for Premium this month! I need a refund immediately.",
         "Hi there! We can definitely look into double charges. Please send us a DM with your Spotify username and billing receipt screenshot. ^BP"),
        ("Billing & Refund Request",
         "@SpotifyCares I canceled my subscription last week but was still charged $10.99 today. Refund please!",
         "Hey! We'd be glad to check your cancellation status. DM us your account email so we can verify the billing cycle details. ^CA"),
        ("Billing & Refund Request",
         "@SpotifyCares Unknown charge from Spotify on my bank statement that I did not authorize.",
         "Hi! We take billing security seriously. Please DM us the date and exact amount charged so we can locate the transaction. ^NK"),
        ("Billing & Refund Request",
         "@SpotifyCares How long does a refund take to reflect back into my Visa credit card?",
         "Hey! Once issued, refunds typically take 3-5 business days to appear on your bank account statement depending on your bank. ^JM"),
        ("Billing & Refund Request",
         "@SpotifyCares Changed my payment card details but the payment keeps failing with payment error.",
         "Hi! Make sure your card is enabled for international online recurring transactions. DM us if you need further help! ^BP"),

        # Playback & Audio Bugs
        ("Playback & Audio Bugs",
         "@SpotifyCares Songs keep pausing automatically every 10 seconds on Android when screen locks.",
         "Hey! Check your battery saver settings for Spotify and set background battery usage to Unrestricted. Let us know if that fixes it! ^CA"),
        ("Playback & Audio Bugs",
         "@SpotifyCares Audio quality sounds crackly and muffled on my Bluetooth headphones on desktop.",
         "Hi! Try toggling Very High audio quality in Settings and turn off Hardware Acceleration under Advanced Settings. ^NK"),
        ("Playback & Audio Bugs",
         "@SpotifyCares Offline downloaded songs won't play when I turn off Wi-Fi/Cellular data.",
         "Hey! Make sure Offline Mode is toggled on in Spotify app settings. Also verify your device connected online within the last 30 days. ^JM"),
        ("Playback & Audio Bugs",
         "@SpotifyCares Crossfade between tracks isn't working on my iOS app after latest update.",
         "Hi! Try toggling Crossfade off and on again under Settings > Playback. Restarting your device can also help. ^BP"),
        ("Playback & Audio Bugs",
         "@SpotifyCares Volume drops dramatically whenever a new song starts playing.",
         "Hey! Check if Enable Audio Normalization is switched on in your Playback settings. Send us a DM if the issue continues! ^CA"),

        # App Crash & Performance
        ("App Crash & Performance",
         "@SpotifyCares The Spotify desktop app crashes every time I open my Saved Songs library.",
         "Hi! This could be caused by corrupted cache. Try performing a clean reinstall of the desktop app. DM us if it keeps crashing! ^NK"),
        ("App Crash & Performance",
         "@SpotifyCares App is super slow and freezing on my iPad Air. Takes 30 seconds to load any page.",
         "Hey! Go to Settings > Storage > Clear Cache. This won't delete your downloaded songs! Let us know if it speeds things up. ^JM"),
        ("App Crash & Performance",
         "@SpotifyCares Spotify crashes immediately on startup after updating to iOS 18.",
         "Hi! Make sure you have the latest Spotify update installed from the App Store. A quick phone restart can also clear temporary glitches. ^BP"),
        ("App Crash & Performance",
         "@SpotifyCares High battery drain on Android after the latest app update, phone gets super hot.",
         "Hey! Send us a DM with your Spotify app version and device model details so our tech team can take a look! ^CA"),
        ("App Crash & Performance",
         "@SpotifyCares Spotify Connect is not finding my Smart TV or Chromecast speakers.",
         "Hi there! Ensure both your mobile phone and Smart TV are connected to the exact same Wi-Fi network. ^NK"),

        # Account Compromise & Security
        ("Account Compromise & Security",
         "@SpotifyCares Someone hacked my account! The email was changed without my permission and my playlists were wiped!",
         "Hi! We're here to help recover your account right away. Please send us a direct message immediately with any original payment receipt or former email. ^JM"),
        ("Account Compromise & Security",
         "@SpotifyCares I see strange songs playing on my account from a device called Living Room PC that I don't own!",
         "Hey! Go to your account page at spotify.com/account and click Sign out everywhere, then change your password right away. DM us if you need help! ^BP"),
        ("Account Compromise & Security",
         "@SpotifyCares Got an email saying my email address was updated on Spotify but I didn't do it!",
         "Hi! That sounds suspicious. Send us a DM immediately with your account username so our security team can lock and restore your account. ^CA"),
        ("Account Compromise & Security",
         "@SpotifyCares Someone added random explicit songs to my private playlists. Is my account compromised?",
         "Hey! We suggest resetting your password and revoking access to 3rd-party apps at spotify.com/account/apps. Send us a DM if you need further security checks! ^NK"),

        # Playlist & Library Issues
        ("Playlist & Library Issues",
         "@SpotifyCares All my saved playlists disappeared after updating the app!",
         "Hi! Make sure you are logged into the exact same account. DM us your username if they're still missing! ^JM"),
        ("Playlist & Library Issues",
         "@SpotifyCares How do I sync local MP3 files from my computer to my iPhone Spotify app?",
         "Hey! Turn on Local Files in app settings on both devices, put both on the same Wi-Fi network, and add the songs to a playlist! ^BP"),
        ("Playlist & Library Issues",
         "@SpotifyCares Is there a limit to how many songs I can add to my Liked Songs library?",
         "Hi! Great news! There is no longer a 10,000 song limit for Liked Songs. You can save as many tracks as you want! ^CA"),
        ("Playlist & Library Issues",
         "@SpotifyCares Accidentally deleted my favorite playlist! Can I recover it?",
         "Hey! Yes! Log in to spotify.com/account and go to Recover Playlists on the left menu. You'll find your deleted playlist there! ^NK"),

        # General Query & Feature Request
        ("General Query & Feature Request",
         "@SpotifyCares When is Spotify HiFi / Lossless audio quality releasing in India?",
         "Hi there! We don't have an exact release date to share right now, but stay tuned to our newsroom for official announcements! ^JM"),
        ("General Query & Feature Request",
         "@SpotifyCares Can I disable podcast recommendations from showing up on my home feed?",
         "Hey! We appreciate the feedback! Currently podcast recommendations are part of the home feed, but we'll share your suggestion with our team. ^BP"),
        ("General Query & Feature Request",
         "@SpotifyCares Is Spotify Wrapped 2026 available yet?",
         "Hi! Spotify Wrapped usually drops towards the end of the year in late November or early December. Stay tuned! ^CA"),
        ("General Query & Feature Request",
         "@SpotifyCares How do I turn on real-time synchronized lyrics on Spotify desktop app?",
         "Hey! Click the small microphone icon in the bottom right playback bar while a song is playing to view live lyrics! ^NK")
    ]

    expanded_rows = []
    pair_id = 1

    for category, customer_msg, support_reply in raw_conversations:
        expanded_rows.append({
            "pair_id": f"SPOT-{pair_id:04d}",
            "intent_category": category,
            "raw_customer_message": customer_msg,
            "raw_support_reply": support_reply
        })
        pair_id += 1

    templates = [
        ("Login & Auth Issue", "Can't sign in on {device}, error says {err}.", "Hi! Try clearing app cache and restarting your device. Send us a DM if issue persists! ^NK"),
        ("Subscription & Premium", "Want to switch my plan from {p1} to {p2}. How does billing work?", "Hey! You can change your plan under account overview. Any unused time will be prorated! ^JM"),
        ("Billing & Refund Request", "Charged {amt} on my card for Spotify but I don't have an active subscription.", "Hi! DM us the charge details and transaction date so we can track down the account responsible. ^BP"),
        ("Playback & Audio Bugs", "Songs stop playing randomly after {time} on {device}.", "Hey! Check background battery optimization settings for Spotify on your device. ^CA"),
        ("App Crash & Performance", "Spotify app keeps crashing when loading {screen} on {device}.", "Hi! Perform a clean reinstall of the app to clear corrupted cache. ^NK"),
        ("Account Compromise & Security", "Got account alert that password was modified on {device} without my consent.", "Hi! Send us a direct message immediately so our security team can secure your account. ^JM"),
        ("Playlist & Library Issues", "Cannot edit or reorder tracks in my playlist {pname} on {device}.", "Hey! Make sure playlist isn't set to read-only or collaborative mode without edit permissions. ^BP"),
        ("General Query & Feature Request", "Is {feat} available in my region yet?", "Hi! Keep an eye on our official newsroom updates for feature availability in your region! ^CA")
    ]

    devices = ["Android 14", "iPhone 15", "MacBook Pro", "Windows 11", "Web Player", "Apple Watch", "iPad Air"]
    errors = ["Error 404", "Authentication Failed", "Offline Mode Stuck", "Session Expired", "Network Timeout"]
    plans = [("Individual", "Family"), ("Student", "Duo"), ("Free", "Individual Premium")]
    amounts = ["$10.99", "$16.99", "$5.99", "$19.99"]
    times = ["30 seconds", "1 minute", "every song transition"]
    screens = ["Liked Songs", "Home Tab", "Search Screen", "Artist Page"]
    pnames = ["My Top 2026 Hits", "Chill Vibes", "Workout Mix", "Roadtrip Songs"]
    feats = ["Audiobook streaming", "Hi-Res Audio", "Canvas Animations", "DJ AI Feature"]

    np.random.seed(42)
    for _ in range(500):
        t = templates[np.random.randint(0, len(templates))]
        cat, msg_tmpl, reply_tmpl = t

        dev = np.random.choice(devices)
        err = np.random.choice(errors)
        p1, p2 = plans[np.random.randint(0, len(plans))]
        amt = np.random.choice(amounts)
        tm = np.random.choice(times)
        scr = np.random.choice(screens)
        pn = np.random.choice(pnames)
        ft = np.random.choice(feats)

        msg = f"@SpotifyCares {msg_tmpl.format(device=dev, err=err, p1=p1, p2=p2, amt=amt, time=tm, screen=scr, pname=pn, feat=ft)}"
        reply = reply_tmpl

        expanded_rows.append({
            "pair_id": f"SPOT-{pair_id:04d}",
            "intent_category": cat,
            "raw_customer_message": msg,
            "raw_support_reply": reply
        })
        pair_id += 1

    df_raw = pd.DataFrame(expanded_rows)
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df_raw.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw dataset: {RAW_DATA_PATH} ({len(df_raw)} records)")

    df_cleaned = df_raw.copy()

    df_cleaned["customer_message"] = (
        df_cleaned["raw_customer_message"].apply(clean_text)
    )

    df_cleaned["support_reply"] = (
        df_cleaned["raw_support_reply"].apply(clean_text)
    )

    df_cleaned = df_cleaned[
        df_cleaned["customer_message"].str.len() >= 10
    ]

    df_cleaned = df_cleaned[
        df_cleaned["support_reply"].str.len() >= 10
    ]

    df_cleaned = df_cleaned.drop_duplicates(
        subset=["customer_message", "support_reply"]
    ).reset_index(drop=True)

    df_cleaned.to_csv(CLEAN_DATA_PATH, index=False)

    print(
        f"Saved cleaned dataset: {CLEAN_DATA_PATH} "
        f"({len(df_cleaned)} records)"
    )

    return df_cleaned

if __name__ == "__main__":
    generate_spotify_dataset()
