import browser_cookie3
import json

cookies = browser_cookie3.chrome()

youtube_cookies = [cookie for cookie in cookies if '.youtube.com' in cookie.domain]

with open('youtube_cookies.txt', 'w') as f:
    for cookie in youtube_cookies:
        f.write(f"{cookie.domain}\tTRUE\t{cookie.path}\t{'TRUE' if cookie.secure else 'FALSE'}\t{cookie.expires}\t{cookie.name}\t{cookie.value}\n")
