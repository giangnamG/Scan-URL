import multiprocessing
import queue
import time
import random
import requests
import urllib.parse
import pandas as pd
import sys
import google.generativeai as genai
import json
import os 

# *** THÊM THƯ VIỆN MÀU SẮC ***
import colorama
from colorama import Fore, Style, Back

# --- Configuration ---

API_KEY_URL_SCAN = os.getenv('URLSCAN_API_KEY', '')
API_KEY_GOOGLE_AI = os.getenv('GOOGLE_API_KEY', '')
CONFIG_FILE = "config.json"
DEFAULT_GOOGLE_AI_MODEL = "gemini-2.5-flash"
DEFAULT_TARGET_DOMAIN = "giaohangtietkiem.vn"
DEFAULT_TARGET_BUSINESS_DESC = "công ty giao hàng/vận chuyển/logistics hàng đầu tại Việt Nam"

# Kiểm tra và cảnh báo nếu chưa đặt API Keys
# *** BỔ SUNG: In cảnh báo nếu key mặc định chưa được thay đổi ***

if 'YOUR_' in API_KEY_URL_SCAN or 'YOUR_' in API_KEY_GOOGLE_AI:
    print(f"{Fore.RED+Style.BRIGHT}API key placeholder detected. Set URLSCAN_API_KEY and GOOGLE_API_KEY, or configure config.json.{Style.RESET_ALL}")
    sys.exit(1)

# Các mẫu pattern để kết hợp với keyword

MATCHES = [
    '.$KEYWORD$',
    '-$KEYWORD$',
    '$KEYWORD$.',
    '$KEYWORD$-',
    '-$KEYWORD$-',
    '.$KEYWORD$.',
    '$KEYWORD$', # Keyword gốc không có ký tự đặc biệt
]

# --- Color Definitions ---
# *** BỔ SUNG: Định nghĩa màu sắc ***

PROCESS_COLORS = {
    "P1": Fore.CYAN,
    "P2": Fore.MAGENTA,
    "P3": Fore.YELLOW,
    "MAIN": Fore.WHITE,
    "HELPER": Fore.BLUE,
    "P2-AI": Fore.LIGHTMAGENTA_EX # Màu riêng cho logic AI trong P2
}
STATUS_COLORS = {
    "SUCCESS": Fore.GREEN + Style.BRIGHT,
    "ERROR": Fore.RED + Style.BRIGHT,
    "WARN": Fore.YELLOW + Style.BRIGHT, # Đổi tên từ WARNING
    "INFO": Fore.WHITE,
    "DEBUG": Fore.LIGHTBLACK_EX,
    "SEND": Fore.BLUE + Style.BRIGHT,
    "RECEIVE": Fore.LIGHTBLUE_EX,
    "WAIT": Fore.LIGHTYELLOW_EX
}

# Kích thước lô để gửi đến AI
BATCH_SIZE = 150 # Điều chỉnh nếu cần
TARGET_BUSINESS_DESC = DEFAULT_TARGET_BUSINESS_DESC # Mô tả target

# --- Logging Function ---
# *** BỔ SUNG: Hàm log với màu sắc ***

def log_message(process_tag, message, level="INFO"):
    """Prints a message with timestamp and colors based on process and level."""
    process_color = PROCESS_COLORS.get(process_tag, Fore.WHITE)
    status_color = STATUS_COLORS.get(level.upper(), STATUS_COLORS["INFO"])
    timestamp = time.strftime('%H:%M:%S')
    # Sử dụng f-string để dễ đọc hơn
    print(f"{Style.DIM}{timestamp}{Style.NORMAL} "
          f"{process_color}[{process_tag}]{Style.RESET_ALL} "
          f"{status_color}{message}{Style.RESET_ALL}")


# --- Helper function to read files ---

def read_lines_from_file(filename):
    """Reads lines from a file, strips whitespace/newlines, and filters empty lines."""
    lines = []
    process_tag = "HELPER" # Tag cho hàm này
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        # *** SỬA: Sử dụng log_message ***
        log_message(process_tag, f"Read {len(lines)} items from '{filename}'", "SUCCESS")
    except FileNotFoundError:
        # *** SỬA: Sử dụng log_message ***
        log_message(process_tag, f"File '{filename}' not found.", "ERROR")
    except Exception as e:
        # *** SỬA: Sử dụng log_message ***
        log_message(process_tag, f"Error reading file '{filename}': {e}", "ERROR")
    return lines

def load_config(config_file=CONFIG_FILE):
    """Loads the JSON config file. Supports the old one-item list format."""
    process_tag = "MAIN"
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except FileNotFoundError:
        log_message(process_tag, f"Config file '{config_file}' not found. Falling back to local text files.", "WARN")
        return {}
    except json.JSONDecodeError as e:
        log_message(process_tag, f"Config file '{config_file}' is invalid JSON: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        log_message(process_tag, f"Error reading config file '{config_file}': {e}", "ERROR")
        sys.exit(1)

    if isinstance(config_data, list):
        if not config_data:
            log_message(process_tag, f"Config file '{config_file}' is an empty list. Falling back to local text files.", "WARN")
            return {}
        log_message(process_tag, f"Config file '{config_file}' uses old list format. Using the first item.", "WARN")
        config_data = config_data[0]

    if not isinstance(config_data, dict):
        log_message(process_tag, f"Config file '{config_file}' must contain a JSON object.", "ERROR")
        sys.exit(1)

    log_message(process_tag, f"Loaded config from '{config_file}'.", "SUCCESS")
    return config_data

def normalize_list(value):
    """Converts config strings/lists into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = value.replace(",", "\n").splitlines()
    else:
        raw_items = [value]
    return [str(item).strip() for item in raw_items if str(item).strip()]

def get_first_config_value(config, keys, default=None):
    """Returns the first scalar value from a list of supported config keys."""
    for key in keys:
        if key in config:
            values = normalize_list(config.get(key))
            if values:
                return values[0]
    return default

def get_config_scalar(config, keys, default=None):
    """Returns a scalar config value without splitting regular prose on commas."""
    for key in keys:
        if key not in config:
            continue
        value = config.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if value not in (None, ""):
            return str(value).strip()
    return default

def get_nested_config(config, section, key, default=None):
    section_value = config.get(section, {})
    if isinstance(section_value, dict):
        value = section_value.get(key)
        if value not in (None, ""):
            return value
    return default

def load_config_list(config, keys, fallback_file=None, label="items"):
    """Loads a list from config first, then from a fallback text file."""
    process_tag = "MAIN"
    for key in keys:
        if key in config:
            values = normalize_list(config.get(key))
            log_message(process_tag, f"Loaded {len(values)} {label} from config key '{key}'.", "SUCCESS")
            return values

    if fallback_file:
        log_message(process_tag, f"No config key found for {label}. Falling back to '{fallback_file}'.", "WARN")
        return read_lines_from_file(fallback_file)

    return []

def resolve_api_key(config, section_key, legacy_keys, env_names, legacy_default=""):
    """Resolves API key from env first, then config, then existing legacy default."""
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value

    api_keys = config.get("api_keys", {})
    if isinstance(api_keys, dict):
        value = api_keys.get(section_key)
        if value:
            return value

    for key in legacy_keys:
        value = config.get(key)
        if value:
            return value

    return legacy_default

def normalize_filter_value(value):
    """Normalizes config/file filter values that may be plain text, domains, or URLs."""
    text = str(value).strip().lower()
    if not text:
        return ""
    parsed = urllib.parse.urlparse(text if "://" in text else f"//{text}")
    host = parsed.netloc or parsed.path.split("/", 1)[0]
    return host.split("@")[-1].split(":", 1)[0].strip(".") or text

def is_whitelisted_domain(domain_lower, whitelist):
    """Returns True only for exact whitelisted domains or their real subdomains."""
    for item in whitelist:
        if domain_lower == item or domain_lower.endswith(f".{item}"):
            return True
    return False

# --- Filtering Logic (Cho Tiến trình 1) ---
# (Hàm isPositiveDomain không cần log màu)
def isPositiveDomain(keyword, seen_domains_global_set, domain, blacklist, whitelist):
    """Checks if a domain is potentially interesting based on various criteria."""
    if not domain: return False
    domain_lower = domain.lower()
    keyword_lower = keyword.lower()
    if any(bad_word in domain_lower for bad_word in blacklist): return False
    if keyword_lower not in domain_lower: return False
    if is_whitelisted_domain(domain_lower, whitelist): return False
    if domain_lower in seen_domains_global_set: return False
    return True

# --- Các hàm cho Tiến trình ---

# ==================================
#        TIẾN TRÌNH 1: CRAWLER
# ==================================

def process_1_crawler(queue_to_ai, keywords, acronyms, blacklist, whitelist, urlscan_api_key):
    """
    Crawls urlscan.io, sending detailed domain dictionaries to P2 after processing EACH PAGE.
    """
    process_tag = "P1"
    log_message(process_tag, "Crawler started.", "INFO")

    KEYWORDS = [item.lower() for item in normalize_list(keywords)]
    ACRONYM = [item.lower() for item in normalize_list(acronyms)]
    BLACKLIST = [normalize_filter_value(item) for item in normalize_list(blacklist)]
    WHITELIST = [normalize_filter_value(item) for item in normalize_list(whitelist)]
    log_message(process_tag, f"Loaded {len(KEYWORDS)} keywords, {len(ACRONYM)} acronyms, {len(BLACKLIST)} blacklist entries, {len(WHITELIST)} whitelist entries.", "INFO")

    if not KEYWORDS and not ACRONYM:
        log_message(process_tag, "No keywords or acronyms loaded. Exiting.", "ERROR")
        queue_to_ai.put(None)
        return

    headers = {
        "Sec-Ch-Ua-Platform": "Windows",
        "X-Requested-With": "XMLHttpRequest",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Sec-Ch-Ua": "\"Not:A-Brand\";v=\"24\", \"Chromium\";v=\"134\"",
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://urlscan.io/search/",
        "Accept-Encoding": "gzip, deflate, br",
        "Priority": "u=1, i",
    }
    if urlscan_api_key:
        headers["API-Key"] = urlscan_api_key
    
    seen_domains_global = set() # Track domains seen across the entire process

    combined_keywords_set = set(KEYWORDS) | set(ACRONYM)
    combined_keywords_list = list(combined_keywords_set)
    acronym_set = set(ACRONYM)

    log_message(process_tag, f"Processing {len(combined_keywords_list)} unique keywords/acronyms...", "INFO")

    total_items_sent = 0 # Đếm tổng số item đã gửi đi

    for keyword in combined_keywords_list:
        is_acronym = keyword in acronym_set
        current_matches = ['$KEYWORD$'] if is_acronym else MATCHES
        log_message(process_tag, f"-- Keyword: '{keyword}' {'(Acronym)' if is_acronym else ''} --", "INFO")
        keyword_total_items_found_this_keyword = 0

        for pattern in current_matches:
            match = pattern.replace("$KEYWORD$", keyword)
            raw_query = f'*page.url.keyword:https\\:*{match}* page.ip:* AND date:>now-9999d'
            encoded_query = urllib.parse.quote(raw_query)
            base_url = f"https://urlscan.io/api/v1/search/?q={encoded_query}"
            log_message(process_tag, f"Querying pattern `{match}`...", "INFO")

            has_more = True
            search_after_param = ""
            page_count = 0
            pattern_total_items_found = 0

            while has_more:
                page_count += 1
                current_url = base_url + search_after_param
                # log_message(process_tag, f"Requesting page {page_count}: {current_url[:80]}...", "DEBUG")

                details_for_this_page = []

                try:
                    response = requests.get(current_url, headers=headers, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    results = data.get("results", [])

                    if not results:
                        if page_count == 1: log_message(process_tag, f"  No results for initial page of pattern `{match}`.", "INFO")
                        has_more = False; break

                    for r in results:
                        page_data = r.get("page", {})
                        task_data = r.get("task", {})
                        domain = page_data.get("domain")

                        if isPositiveDomain(keyword, seen_domains_global, domain, BLACKLIST, WHITELIST):
                            task_url = page_data.get("url", task_data.get("url", ""))
                            title = page_data.get("title", "")
                            ip = page_data.get("ip", "")
                            uuid = task_data.get("uuid", "")
                            server = page_data.get("server", "")
                            asn = page_data.get("asnname", "")
                            result_link = r.get("result", "")
                            status = str(page_data.get("status", ""))

                            result_details = { # Create dict
                                "Domain": domain, "URL": task_url, "Title": title,
                                "IP Address": ip, "ASN Name": asn, "Server": server,
                                "Status": status, "Scan UUID": uuid, "urlscan.io Result": result_link,
                                "Matched Keyword": keyword, "Matched Pattern": match
                            }
                            details_for_this_page.append(result_details)
                            seen_domains_global.add(domain.lower())

                    if details_for_this_page:
                        page_item_count = len(details_for_this_page)
                        pattern_total_items_found += page_item_count
                        total_items_sent += page_item_count
                        log_message(process_tag, f"  Page {page_count}: Found {page_item_count} new items. Sending batch to AI Process...", "SEND")
                        queue_to_ai.put(details_for_this_page)

                    last_sort = results[-1].get("sort")
                    if last_sort and isinstance(last_sort, list) and len(last_sort) > 0:
                        search_after = ",".join(map(str, last_sort))
                        search_after_param = f"&search_after={search_after}"
                        time.sleep(random.uniform(1.2, 2.8))
                    else:
                        has_more = False

                except requests.exceptions.Timeout:
                    log_message(process_tag, f"⏱️ Timeout querying page {page_count} for pattern `{match}`. Moving to next pattern.", "WARN")
                    has_more = False; time.sleep(10)
                except requests.exceptions.RequestException as e:
                    status_code = e.response.status_code if e.response is not None else 'N/A'
                    log_message(process_tag, f"❌ Request Error querying page {page_count} for pattern `{match}`: {e}. Status: {status_code}. Moving to next pattern.", "ERROR")
                    has_more = False; time.sleep(5)
                except json.JSONDecodeError as e:
                    log_message(process_tag, f"❌ JSON Decode Error querying page {page_count} for pattern `{match}`: {e}. Moving to next pattern.", "ERROR")
                    has_more = False; time.sleep(5)
                except Exception as e:
                    log_message(process_tag, f"❌ Unexpected Error querying page {page_count} for pattern `{match}`: {e}. Moving to next pattern.", "ERROR")
                    has_more = False; time.sleep(5)

            if pattern_total_items_found > 0:
                log_message(process_tag, f"  Finished pattern `{match}`. Total new items sent for this pattern: {pattern_total_items_found}", "INFO")
                keyword_total_items_found_this_keyword += pattern_total_items_found
            else:
                log_message(process_tag, f"  Finished pattern `{match}`. No new domains found for this pattern.", "INFO")

            time.sleep(random.uniform(1.8, 3.5))

        log_message(process_tag, f"-- Finished Keyword: '{keyword}'. Total items found: {keyword_total_items_found_this_keyword} --", "SUCCESS")

    log_message(process_tag, f"Crawler finished all keywords. Total unique items sent: {total_items_sent}", "SUCCESS")
    queue_to_ai.put(None)
    log_message(process_tag, "Sent termination signal.", "SEND")


# ==================================
#     TIẾN TRÌNH 2: AI ANALYZER
# ==================================

class GoogleAI_Processor:
    """Handles interaction with the Google AI API."""
    def __init__(self, api_key, model_name=DEFAULT_GOOGLE_AI_MODEL):
        self.model = None
        self.is_configured = False
        self.model_name = model_name or DEFAULT_GOOGLE_AI_MODEL
        self.last_review_status = "not_started"
        self.last_error_message = ""
        process_tag = "P2-AI"
        if not api_key or "YOUR_" in api_key:
            self.last_review_status = "skipped_not_configured"
            self.last_error_message = "Google AI API key is missing."
            log_message(process_tag, "Google AI API key is missing. AI analysis will be skipped.", "WARN")
            return
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.is_configured = True
            log_message(process_tag, f"Google AI configured successfully with model '{self.model_name}'.", "SUCCESS") # Sử dụng log_message
        except Exception as e:
            self.last_review_status = "skipped_not_configured"
            self.last_error_message = str(e)
            log_message(process_tag, f"Error configuring Google AI: {e}", "ERROR") # Sử dụng log_message
            log_message(process_tag, "AI analysis will be skipped.", "WARN")       # Sử dụng log_message

    # Bên trong class GoogleAI_Processor
    def get_prompt(self, target_domain, domain_list_str):
        """Generates the optimized and detailed prompt for the AI."""
        return f"""
            Bạn là một hệ thống tự động chuyên biệt, được huấn luyện để phân tích và lọc danh sách tên miền/URL nhằm phát hiện các trường hợp có khả năng giả mạo hoặc gây nhầm lẫn với một tên miền mục tiêu cụ thể. Nhiệm vụ của bạn là áp dụng các tiêu chí một cách nhất quán và chỉ trả về kết quả theo định dạng yêu cầu.

            **Tên miền mục tiêu chính thức cần bảo vệ:** "{target_domain}"

            **Danh sách cần kiểm tra:**
            ```
            {domain_list_str}
            ```

            **Nhiệm vụ:**
            Phân tích kỹ lưỡng danh sách trên. Xác định và trích xuất **chỉ** những mục (tên miền hoặc URL đầy đủ như trong danh sách) có dấu hiệu đáng ngờ là đang cố gắng giả mạo hoặc gây nhầm lẫn với "{target_domain}". Áp dụng các tiêu chí sau:

            **Các Tiêu chí Xác định Đáng ngờ (Ưu tiên xem xét):**
            1.  **Typosquatting/Lỗi chính tả:** Tên miền chứa lỗi chính tả nhỏ so với "{target_domain}". Ví dụ cụ thể: thay `t` bằng `l`, thay `I` (hoa) bằng `l` (thường), thay `O` (hoa) bằng `0` (số không), thêm/bớt một vài ký tự, đảo vị trí các ký tự gần nhau (ví dụ: `giaohangtietkeim.vn`).
            2.  **Thêm Từ khóa Đáng ngờ:** Tên miền kết hợp phần chính của "{target_domain}" (hoặc các biến thể viết tắt phổ biến như 'ghtk') với các từ khóa nhạy cảm hoặc thường dùng trong lừa đảo. Ví dụ từ khóa: `login`, `signin`, `dangnhap`, `taikhoan`, `support`, `help`, `payment`, `thanhtoan`, `admin`, `portal`, `service`, `vn`, `247`, `online`, `chat`, `hotro`, `tracuu`, `kiemtra`, `donhang`, `shippervn`, `shipnhanhvn`, `khieunai`, `cskh`. Ví dụ tên miền: `login-giaohangtietkiem.com`, `hotro-ghtk.net`, `tracuu-giaohangtietkiem.xyz`, `giaohangtietkiem247.top`, `chat.giaohangtietkiem.co`.
            3.  **Hoán vị/Tái cấu trúc Gây nhầm lẫn:** Sử dụng các phần của tên miền "{target_domain}" (hoặc biến thể 'ghtk') kết hợp với các từ khác (có thể liên quan hoặc không) theo cấu trúc hoặc thứ tự khác biệt nhằm gây nhầm lẫn. Ví dụ cụ thể cho "{target_domain}": `vn-ghtk.com`, `giaohang-tk.org`, `hotro-ghtk.info`, `chat-giaohangtk.xyz`, `ghtk-donhang.net`.
            4.  **Tên miền phụ (Subdomain) Đáng ngờ:** Sử dụng tên miền phụ trông giống hoặc liên quan đến "{target_domain}" trên một tên miền gốc *không liên quan hoặc đáng ngờ*. Ví dụ: `giaohangtietkiem.vn.maliciousdomain.com`, `ghtk.service-provider.net`, `dangnhap.giaohangtietkiem.some-random-site.info`, `congtygiao.congtygiaohangtietkiemvn.xyz`.
            5.  **TLD (Tên miền cấp cao) Không phù hợp/Rủi ro cao:** Sử dụng TLD khác với TLD chính thức của mục tiêu (thường là .vn hoặc .com) mà có thể gây nhầm lẫn hoặc thường liên quan đến các trang độc hại. Ví dụ TLD đáng ngờ: `.info`, `.xyz`, `.cc`, `.co`, `.tk`, `.pw`, `.top`, `.online`, `.website`, `.click`, `.site`, `.shop`, `.club`. Ví dụ tên miền: `giaohangtietkiem.info`, `ghtk.xyz`, `dangnhapghtk.co`.
            6.  **Sử dụng Dấu gạch ngang Bất thường:** Chèn dấu gạch ngang (-) ở những vị trí không tự nhiên hoặc quá nhiều để cố gắng làm giống tên miền gốc hoặc tạo biến thể mới. Ví dụ: `giao-hang-tiet-kiem.com`, `ghtk-login-portal.net`.

            **Các Tiêu chí Loại trừ Quan trọng (KHÔNG coi là đáng ngờ nếu chỉ dựa vào các điểm này):**
            *   **Hoàn toàn Không liên quan:** Các tên miền không chứa bất kỳ phần nào giống hoặc liên quan đến "{target_domain}" hoặc các biến thể của nó. Ví dụ: `google.com`, `facebook.com`, `nightking-studio.itch.io`.
            *   **Dịch vụ Hợp pháp:** Các URL chứa tên miền mục tiêu nhưng đến từ các dịch vụ tracking, marketing, hoặc rút gọn link uy tín và đã biết (ví dụ: URL dài từ `sendgrid.net`, `hubspot.com`, `bit.ly` *trỏ đến* tên miền mục tiêu - nhưng bản thân domain của dịch vụ đó thì không đáng ngờ). *Lưu ý: Vẫn cần cảnh giác nếu đường dẫn URL trông đáng ngờ.*
            *   **Trùng khớp Ngẫu nhiên/Ngữ nghĩa Khác biệt:** Các tên miền/URL có thể chứa từ viết tắt (như 'ghtk') nhưng rõ ràng thuộc về một thương hiệu, dịch vụ hoặc ngữ cảnh hoàn toàn khác, không liên quan đến lĩnh vực hoạt động của "{target_domain}". Ví dụ cụ thể cần LOẠI TRỪ: `hermes.nightkey.com`, `ias-maverickcap.lightkeeperhq.com`, `n6qqi-oaaaa-aaaad-qed4a-cai.icp0.io/posts/fault-stp-lightkravte/index.html`.
            *   **Trường hợp KHÔNG LOẠI TRỪ (Vẫn cần xem xét là đáng ngờ):** Nếu tên miền chứa từ viết tắt nhưng *toàn bộ tên miền* hoặc ngữ cảnh của nó *có vẻ* liên quan đến lĩnh vực của "{target_domain}", hãy vẫn coi là đáng ngờ. Ví dụ: `huythehoivienghtk247.net` (có 'ghtk' và '247', liên quan đến dịch vụ, nên đưa vào).

            **YÊU CẦU ĐỊNH DẠNG ĐẦU RA TUYỆT ĐỐI NGHIÊM NGẶT:**
            1.  Toàn bộ phản hồi của bạn **PHẢI** là một **danh sách JSON (JSON array)** hợp lệ chứa các chuỗi (strings).
            2.  Mỗi chuỗi trong danh sách là một URL hoặc tên miền từ danh sách đầu vào được xác định là đáng ngờ theo các tiêu chí trên.
            3.  **VÍ DỤ ĐÚNG:** `["suspicious1.com/path", "suspicious2.net", "suspicious3.info/login"]`
            4.  **VÍ DỤ ĐÚNG (Không tìm thấy):** `[]` (Một danh sách JSON rỗng).
            5.  **TUYỆT ĐỐI KHÔNG:**
                *   KHÔNG trả về đối tượng JSON (object/dictionary). Ví dụ sai: `{{"suspicious_domains": [...]}}` hoặc `{{"analysis_results": [...]}}`.
                *   KHÔNG bao gồm bất kỳ văn bản giải thích, phân tích, lý do, lời chào, mô tả, hay ký tự nào khác ngoài danh sách JSON.
                *   KHÔNG sử dụng định dạng markdown.
            6.  Phản hồi phải bắt đầu bằng `[` và kết thúc bằng `]`.

            **Lý do yêu cầu nghiêm ngặt:** Hệ thống tự động phía sau sẽ trực tiếp phân tích cú pháp phản hồi này và chỉ chấp nhận định dạng danh sách JSON thuần túy. Bất kỳ sai lệch nào sẽ gây lỗi nghiêm trọng. Hãy đảm bảo tuân thủ tuyệt đối.
            """

    def analyze_domains(self, target_domain, domain_list):
        """Analyzes a list of domain names using the Google AI model."""
        process_tag = "P2-AI" # Tag cho các log bên trong hàm này
        self.last_review_status = "review_failed"
        self.last_error_message = ""
        if not self.is_configured or not self.model:
            self.last_review_status = "skipped_not_configured"
            self.last_error_message = "AI model unavailable."
            log_message(process_tag, "AI Model unavailable. Skipping analysis.", "WARN")
            return []
        if not domain_list:
            self.last_review_status = "skipped_no_input"
            return []

        log_message(process_tag, f"Analyzing batch of {len(domain_list)} domains for target '{target_domain}'...", "INFO")
        domain_list_str = "\n".join(domain_list)
        prompt_text = self.get_prompt(target_domain, domain_list_str)
        suspicious_domains_list = []
        response_text = ""
        retries = 2
        wait_time = 65

        for attempt in range(retries + 1):
            try:
                generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
                response = self.model.generate_content(prompt_text, generation_config=generation_config)
                response_text = response.text
                result_data = json.loads(response_text)

                if isinstance(result_data, list):
                     suspicious_domains_list = result_data
                     self.last_review_status = "reviewed"
                     self.last_error_message = ""
                     log_message(process_tag, "  ✔️ OK: AI returned a direct list.", "SUCCESS")
                     break
                elif isinstance(result_data, dict):
                    log_message(process_tag, "  INFO: AI returned a dictionary. Attempting extraction...", "INFO")
                    extracted = False
                    keys_to_check = ["suspicious_domains", "suspiciousDomains", "subdomains_to_check"]
                    analysis_key = "analysis_results"
                    domain_keys_in_analysis = ["subdomain", "domain"]
                    for key in keys_to_check:
                        if key in result_data and isinstance(result_data[key], list):
                            suspicious_domains_list = result_data[key]
                            self.last_review_status = "reviewed"
                            self.last_error_message = ""
                            log_message(process_tag, f"  ✔️ Extracted list from '{key}' key.", "SUCCESS")
                            extracted = True; break
                    if not extracted and analysis_key in result_data and isinstance(result_data[analysis_key], list):
                        temp_list = []
                        for item in result_data[analysis_key]:
                            if isinstance(item, dict):
                                for d_key in domain_keys_in_analysis:
                                    if d_key in item: temp_list.append(item[d_key]); break
                        if temp_list:
                            suspicious_domains_list = temp_list
                            self.last_review_status = "reviewed"
                            self.last_error_message = ""
                            log_message(process_tag, f"  ✔️ Extracted list from items within '{analysis_key}'.", "SUCCESS")
                            extracted = True
                        else: log_message(process_tag, f"  ⚠️ Found '{analysis_key}' but failed to extract domains.", "WARN")
                    if not extracted:
                        self.last_review_status = "review_failed"
                        self.last_error_message = "AI response did not contain an extractable domain list."
                        log_message(process_tag, "  ⚠️ Couldn't find/extract list under known structures.", "WARN")
                    break
                else:
                    self.last_review_status = "review_failed"
                    self.last_error_message = f"Unexpected AI response type: {type(result_data)}"
                    log_message(process_tag, f"  ⚠️ AI response was not list/dict. Type: {type(result_data)}", "WARN")
                    break
            except json.JSONDecodeError as e:
                self.last_review_status = "review_failed"
                self.last_error_message = f"JSON decode error: {e}"
                log_message(process_tag, f"  ❌ Error decoding JSON (Attempt {attempt + 1}).", "ERROR")
                break
            except Exception as e:
                error_message = str(e)
                self.last_review_status = "review_failed"
                self.last_error_message = error_message
                log_message(process_tag, f"  ❌ Error calling API (Attempt {attempt + 1}): {error_message[:100]}...", "ERROR")
                if ("429" in error_message or "Resource has been exhausted" in error_message or
                    "503" in error_message or isinstance(e, requests.exceptions.Timeout)):
                    if attempt < retries:
                        log_message(process_tag, f"  Retrying after {int(wait_time)} seconds...", "WAIT")
                        time.sleep(wait_time)
                        wait_time *= 1.5
                    else: log_message(process_tag, "  Max retries reached.", "ERROR"); break
                else: log_message(process_tag, "  Unrecoverable API error.", "ERROR"); break
            time.sleep(random.uniform(0.2, 0.5))

        if suspicious_domains_list:
            log_message(process_tag, f"  ➡️ Final extracted suspicious: {len(suspicious_domains_list)} domains.", "INFO")
        return suspicious_domains_list


def process_2_ai_analyzer(queue_from_crawler, queue_to_exporter, target_domain_for_ai, google_ai_api_key, google_ai_model):
    process_tag = "P2"
    log_message(process_tag, "AI Analyzer started.", "INFO")
    ai_processor = GoogleAI_Processor(google_ai_api_key, google_ai_model)
    if not ai_processor.is_configured:
         log_message(process_tag, "AI Processor failed to configure. Analysis will be skipped.", "WARN")

    while True:
        try:
            details_batch = queue_from_crawler.get()
            if details_batch is None:
                log_message(process_tag, "Received termination signal from Crawler.", "RECEIVE")
                break

            if isinstance(details_batch, list) and details_batch:
                log_message(process_tag, f"Received batch of {len(details_batch)} item details from Crawler.", "RECEIVE")
                domain_names_to_analyze = list(set(filter(None, [item.get("Domain") for item in details_batch])))

                suspicious_names_list = []
                ai_review_status = "review_failed"
                ai_review_error = ""

                if not domain_names_to_analyze:
                     ai_review_status = "skipped_no_domain"
                     ai_review_error = "Batch contains no valid domain names to analyze."
                     log_message(process_tag, f"  {ai_review_error}", "WARN")
                elif ai_processor.is_configured:
                    suspicious_names_list = ai_processor.analyze_domains(target_domain_for_ai, domain_names_to_analyze)
                    ai_review_status = ai_processor.last_review_status
                    ai_review_error = ai_processor.last_error_message
                else:
                     ai_review_status = "skipped_not_configured"
                     ai_review_error = ai_processor.last_error_message or "AI processor is not configured."
                     log_message(process_tag, "  Skipping AI analysis (processor not configured).", "INFO")

                suspicious_names_set = {
                    normalize_filter_value(name) for name in suspicious_names_list
                    if normalize_filter_value(name)
                }
                enriched_batch = []
                suspicious_count = 0
                for item in details_batch:
                    enriched_item = item.copy()
                    domain_key = normalize_filter_value(enriched_item.get("Domain", ""))
                    is_suspicious = ai_review_status == "reviewed" and domain_key in suspicious_names_set
                    if is_suspicious:
                        suspicious_count += 1
                    enriched_item["AI_Suspicious"] = is_suspicious
                    enriched_item["AI Review Status"] = ai_review_status
                    enriched_item["AI Review Error"] = ai_review_error
                    enriched_item["AI Model"] = ai_processor.model_name
                    enriched_batch.append(enriched_item)

                if ai_review_status == "reviewed":
                    if suspicious_count:
                        log_message(process_tag, f"  AI reviewed batch. Marked {suspicious_count}/{len(enriched_batch)} items suspicious.", "INFO")
                    else:
                        log_message(process_tag, "  AI reviewed batch. No suspicious domains identified.", "INFO")
                else:
                    log_message(process_tag, f"  AI did not review this batch ({ai_review_status}). Logging {len(enriched_batch)} items anyway.", "WARN")

                log_message(process_tag, f"  Sending {len(enriched_batch)} logged items to Exporter.", "SEND")
                queue_to_exporter.put(enriched_batch)

            elif isinstance(details_batch, list) and not details_batch:
                 log_message(process_tag, "  Received an empty details batch from Crawler.", "INFO")
            else:
                 log_message(process_tag, f"  ⚠️ Warning: Received unexpected data type from crawler queue: {type(details_batch)}", "WARN")

            time.sleep(random.uniform(0.5, 1.5))
        except Exception as e:
            log_message(process_tag, f"Unexpected Error in AI Analyzer main loop: {e}", "ERROR")
            time.sleep(5)

    log_message(process_tag, "AI Analyzer finished processing.", "SUCCESS")
    queue_to_exporter.put(None)
    log_message(process_tag, "Sent termination signal.", "SEND")


# ==================================
#      TIẾN TRÌNH 3: EXPORTER
# ==================================
def process_3_exporter(queue_from_ai, output_filename): # Bỏ target_domain
    process_tag = "P3"; log_message(process_tag, "Exporter started.", "INFO"); all_processed_details = []; received_termination_signal = False
    log_message(process_tag, "Waiting for enriched domain details...", "WAIT")
    while not received_termination_signal:
        try:
            enriched_batch = queue_from_ai.get(timeout=10)
            if enriched_batch is None: log_message(process_tag, "Received termination signal.", "RECEIVE"); received_termination_signal = True
            elif isinstance(enriched_batch, list) and enriched_batch:
                added_count = len(enriched_batch); all_processed_details.extend(enriched_batch)
                log_message(process_tag, f"  Received {added_count} enriched items. Total: {len(all_processed_details)}", "RECEIVE")
            elif isinstance(enriched_batch, list) and not enriched_batch: log_message(process_tag, "  Received empty batch.", "INFO")
            else: log_message(process_tag, f"  ⚠️ Received unexpected type: {type(enriched_batch)}", "WARN")
        except queue.Empty: log_message(process_tag, "Queue empty for 10s...", "DEBUG"); continue
        except Exception as e: log_message(process_tag, f"Exporter loop error: {e}", "ERROR"); time.sleep(5)

    log_message(process_tag, f"Processing {len(all_processed_details)} total collected items for final export.", "INFO")

    if all_processed_details:
        # 1. Loại bỏ trùng lặp dựa trên Domain (giữ lại entry cuối cùng)
        unique_final_details_dict = {}
        for item in all_processed_details:
            domain = item.get("Domain")
            if domain: unique_final_details_dict[domain] = item
        unique_final_details = list(unique_final_details_dict.values())
        if len(unique_final_details) < len(all_processed_details):
            log_message(process_tag, f"Removed {len(all_processed_details) - len(unique_final_details)} duplicate domain entries.", "INFO")

        # 2. Tạo DataFrame đầy đủ (processed)
        try:
            log_message(process_tag, "Creating main DataFrame with processed data...", "INFO");
            df_processed = pd.DataFrame(unique_final_details)

            # 3. Chuẩn bị dữ liệu cho Sheet 1 ('Raw Data')
            df_raw = df_processed.copy()
            # Sắp xếp sheet raw theo Domain nếu muốn
            if 'Domain' in df_raw.columns:
                log_message(process_tag, "Sorting 'Raw Data' sheet by Domain...", "INFO")
                df_raw_sorted = df_raw.sort_values(by='Domain', ascending=True, ignore_index=True, kind='stable')
            else:
                df_raw_sorted = df_raw

            df_ai_suspicious_sorted = pd.DataFrame()
            if 'AI_Suspicious' in df_raw.columns:
                df_ai_suspicious = df_raw[df_raw['AI_Suspicious'] == True].copy()
                if 'Domain' in df_ai_suspicious.columns:
                    df_ai_suspicious_sorted = df_ai_suspicious.sort_values(by='Domain', ascending=True, ignore_index=True, kind='stable')
                else:
                    df_ai_suspicious_sorted = df_ai_suspicious
                log_message(process_tag, f"Created 'AI Suspicious' DataFrame with {len(df_ai_suspicious_sorted)} items.", "INFO")

            df_ai_unreviewed_sorted = pd.DataFrame()
            if 'AI Review Status' in df_raw.columns:
                df_ai_unreviewed = df_raw[df_raw['AI Review Status'].astype(str).str.lower() != 'reviewed'].copy()
                if 'Domain' in df_ai_unreviewed.columns:
                    df_ai_unreviewed_sorted = df_ai_unreviewed.sort_values(by='Domain', ascending=True, ignore_index=True, kind='stable')
                else:
                    df_ai_unreviewed_sorted = df_ai_unreviewed
                log_message(process_tag, f"Created 'AI Unreviewed' DataFrame with {len(df_ai_unreviewed_sorted)} items.", "INFO")

            # 4. Chuẩn bị dữ liệu cho Sheet 2 ('Non-Cloudflare Server')
            df_non_cloudflare = pd.DataFrame() # Khởi tạo rỗng
            if 'Server' in df_raw.columns: # Kiểm tra xem cột Server có tồn tại không
                # Lọc: Server không phải NaN VÀ server (lowercase) không chứa 'cloudflare'
                df_non_cloudflare = df_raw[
                    df_raw['Server'].notna() & \
                    (~df_raw['Server'].astype(str).str.lower().str.contains('cloudflare', na=False))
                ].copy()
                 # Sắp xếp sheet non-cloudflare theo Domain nếu muốn
                if 'Domain' in df_non_cloudflare.columns:
                     log_message(process_tag, "Sorting 'Non-Cloudflare Server' sheet by Domain...", "INFO")
                     df_non_cloudflare_sorted = df_non_cloudflare.sort_values(by='Domain', ascending=True, ignore_index=True, kind='stable')
                else:
                     df_non_cloudflare_sorted = df_non_cloudflare
                log_message(process_tag, f"Created 'Non-Cloudflare Server' DataFrame with {len(df_non_cloudflare_sorted)} items.", "INFO")
            else:
                log_message(process_tag, "'Server' column not found in data, cannot create 'Non-Cloudflare Server' sheet.", "WARN")
                df_non_cloudflare_sorted = df_non_cloudflare # Vẫn là DataFrame rỗng

            # 5. Ghi vào file Excel với nhiều sheet
            log_message(process_tag, f"Writing multiple sheets to '{output_filename}'...", "INFO")
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                 try: os.makedirs(output_dir); log_message(process_tag, f"Created dir: '{output_dir}'", "INFO")
                 except OSError as ose: log_message(process_tag, f"Cannot create dir '{output_dir}': {ose}", "ERROR"); output_filename = os.path.basename(output_filename); log_message(process_tag, f"Writing to current dir: '{output_filename}'", "WARN")

            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                # Ghi Sheet 1 ('Raw Data')
                df_raw_sorted.to_excel(writer, sheet_name='Raw Data', index=False)
                log_message(process_tag, f"  Sheet 'Raw Data' written ({len(df_raw_sorted)} rows).", "INFO")

                if not df_ai_suspicious_sorted.empty:
                    df_ai_suspicious_sorted.to_excel(writer, sheet_name='AI Suspicious', index=False)
                    log_message(process_tag, f"  Sheet 'AI Suspicious' written ({len(df_ai_suspicious_sorted)} rows).", "INFO")
                else:
                    log_message(process_tag, "  Sheet 'AI Suspicious' skipped (no suspicious AI results).", "INFO")

                if not df_ai_unreviewed_sorted.empty:
                    df_ai_unreviewed_sorted.to_excel(writer, sheet_name='AI Unreviewed', index=False)
                    log_message(process_tag, f"  Sheet 'AI Unreviewed' written ({len(df_ai_unreviewed_sorted)} rows).", "INFO")
                else:
                    log_message(process_tag, "  Sheet 'AI Unreviewed' skipped (all logged domains were reviewed by AI).", "INFO")

                # Ghi Sheet 2 ('Non-Cloudflare Server'), chỉ khi có dữ liệu
                if not df_non_cloudflare_sorted.empty:
                    df_non_cloudflare_sorted.to_excel(writer, sheet_name='Non-Cloudflare Servers', index=False)
                    log_message(process_tag, f"  Sheet 'Non-Cloudflare Servers' written ({len(df_non_cloudflare_sorted)} rows).", "INFO")
                else:
                     log_message(process_tag, "  Sheet 'Non-Cloudflare Servers' skipped (no data or 'Server' column missing).", "INFO")

            log_message(process_tag, f"Export successful to '{output_filename}' with multiple sheets.", "SUCCESS")

        except ImportError: log_message(process_tag, "Error: 'pandas'/'openpyxl' needed.", "ERROR")
        except KeyError as e: log_message(process_tag, f"Export Error: Missing key {e}.", "ERROR")
        except Exception as e: log_message(process_tag, f"Export Error: {e}", "ERROR")
    else: log_message(process_tag, "No items processed to export.", "INFO")
    log_message(process_tag, "Exporter finished.", "SUCCESS")


def finnaly_analyer():
    all_domain_title_list = []
    original_data_df = None # Lưu lại DataFrame gốc để lấy chi tiết sau
    try:
        # Đọc sheet đầu tiên của file Excel
        # Giả định sheet đầu tiên là sheet chứa dữ liệu đầy đủ (có thể là 'Raw Data' hoặc 'All Analyzed')
        df = pd.read_excel(INPUT_EXCEL_FILE, sheet_name=0) # Đọc sheet đầu tiên theo index
        original_data_df = df.copy() # Lưu lại để tham chiếu
        log_message(f"Successfully loaded Excel file. Found {len(df)} rows.", "SUCCESS")

        if 'Domain' not in df.columns or 'Title' not in df.columns:
            log_message("Error: Required columns 'Domain' and/or 'Title' not found.", "ERROR")
            sys.exit(1)

        # Trích xuất dữ liệu Domain và Title
        for index, row in df.iterrows():
            domain = row['Domain']
            title = row['Title']
            if isinstance(domain, str) and domain.strip():
                all_domain_title_list.append({
                    "Domain": domain.strip(),
                    "Title": str(title).strip() if pd.notna(title) else ""
                })
        log_message(f"Extracted {len(all_domain_title_list)} valid Domain/Title pairs for AI analysis.", "INFO")

    except FileNotFoundError: log_message(f"Error: Input file '{INPUT_EXCEL_FILE}' not found.", "ERROR"); sys.exit(1)
    except Exception as e: log_message(f"Error reading/processing Excel file: {e}", "ERROR"); sys.exit(1)

    # --- Phân tích bằng AI theo lô ---
    if all_domain_title_list:
        ai_processor = GoogleAI_Processor(API_KEY_GOOGLE_AI)
        final_ai_suspicious_domains = []

        if ai_processor.is_configured:
            log_message("Starting Final AI analysis in batches...", "INFO")
            total_batches = (len(all_domain_title_list) + BATCH_SIZE - 1) // BATCH_SIZE

            read_lines_from_file(acronym_file)
            for i in range(0, len(all_domain_title_list), BATCH_SIZE):
                batch_number = (i // BATCH_SIZE) + 1
                current_batch = all_domain_title_list[i:i + BATCH_SIZE]
                log_message(f"--- Processing Final AI Batch {batch_number}/{total_batches} ({len(current_batch)} items) ---", "INFO")

                # Gọi AI để phân tích lô hiện tại (không cần trusted_examples ở bước này)
                suspicious_in_batch = ai_processor.analyze_domains(
                    TARGET_DOMAIN_AI,
                    TARGET_BUSINESS_DESC,
                    TARGET_ACRONYMS_LIST,
                    current_batch
                )
                final_ai_suspicious_domains.extend(suspicious_in_batch)

                if batch_number < total_batches:
                     sleep_time = random.uniform(1.0, 3.0)
                     log_message(f"--- Sleeping for {sleep_time:.1f}s before next batch ---", "WAIT")
                     time.sleep(sleep_time)

            # Loại bỏ trùng lặp và sắp xếp
            final_suspicious_list = sorted(list(set(final_ai_suspicious_domains)))

            log_message(f"\n=== Final AI Analysis Complete ===", "SUCCESS")
            log_message(f"Total suspicious domains identified by final AI pass: {len(final_suspicious_list)}", "SUCCESS")

            # In kết quả
            if final_suspicious_list:
                print("\n--- Final AI-Identified Suspicious Domains ---")
                # Optional: Lấy lại chi tiết từ DataFrame gốc để in thêm thông tin
                if original_data_df is not None:
                    final_details_df = original_data_df[original_data_df['Domain'].isin(final_suspicious_list)]
                    # Sắp xếp lại theo list cuối cùng nếu muốn giữ thứ tự AI trả về (hoặc sắp xếp theo cột khác)
                    # Hoặc chỉ in list domain là đủ
                    print(final_details_df[['Domain', 'Title', 'Risk Score', 'AI_Suspicious']].to_string(index=False)) # In các cột chính
                else:
                     for domain in final_suspicious_list: print(domain) # In list nếu không lấy được details
                print("------------------------------------------")
            else:
                print("\nNo suspicious domains were identified by the final AI pass.")

        else: log_message("AI Analysis skipped (processor not configured).", "WARN")
    else: log_message("No valid data extracted from Excel to analyze.", "WARN")

    log_message("Script finished.", "INFO")

# --- Hàm chính để quản lý tiến trình ---
if __name__ == "__main__":
    # ... (Giữ nguyên hàm main) ...
    colorama.init(autoreset=True); multiprocessing.freeze_support(); main_tag = "MAIN"
    log_message(main_tag,"="*44, "INFO"); log_message(main_tag,"=== URLScan & AI Phishing Domain Hunter ===", "INFO"); log_message(main_tag,"="*44, "INFO")
    CONFIG = load_config(CONFIG_FILE)
    KEYWORDS = load_config_list(CONFIG, ("keywords", "keyword"), "KEYWORDS", "keywords")
    ACRONYM = load_config_list(CONFIG, ("acronyms", "acronym"), "ACRONYM", "acronyms")
    BLACKLIST = load_config_list(CONFIG, ("blacklist", "backlist"), "BLACKLIST", "blacklist entries")
    WHITELIST = load_config_list(CONFIG, ("whitelist",), "WHITELIST", "whitelist entries")
    TARGET_DOMAIN_AI = get_first_config_value(CONFIG, ("target_domain", "target_domains", "target"), DEFAULT_TARGET_DOMAIN)
    TARGET_BUSINESS_DESC = get_config_scalar(CONFIG, ("target_description", "description"), DEFAULT_TARGET_BUSINESS_DESC)
    GOOGLE_AI_MODEL = str(get_nested_config(CONFIG, "google_ai", "model", CONFIG.get("google_ai_model", DEFAULT_GOOGLE_AI_MODEL))).strip() or DEFAULT_GOOGLE_AI_MODEL
    API_KEY_URL_SCAN = resolve_api_key(CONFIG, "urlscan", ("urlscan_api_key",), ("URLSCAN_API_KEY",), API_KEY_URL_SCAN)
    API_KEY_GOOGLE_AI = resolve_api_key(CONFIG, "google_ai", ("google_api_key", "gemini_api_key"), ("GOOGLE_API_KEY", "GEMINI_API_KEY"), API_KEY_GOOGLE_AI)
    configured_batch_size = get_nested_config(CONFIG, "google_ai", "batch_size", CONFIG.get("batch_size", BATCH_SIZE))
    try:
        BATCH_SIZE = int(configured_batch_size)
    except (TypeError, ValueError):
        log_message(main_tag, f"Invalid batch size '{configured_batch_size}'. Using {BATCH_SIZE}.", "WARN")

    TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
    report_config = CONFIG.get("reports", {}) if isinstance(CONFIG.get("reports", {}), dict) else {}
    REPORTS_DIR = report_config.get("directory", CONFIG.get("reports_dir", "reports"))
    report_prefix = report_config.get("filename_prefix", "domain_report")
    # Đổi tên file output để phản ánh nội dung mới
    output_base_name = f"{report_prefix}_{TARGET_DOMAIN_AI.replace('.', '_')}_{TIMESTAMP}.xlsx"
    OUTPUT_EXCEL_FILE = os.path.join(REPORTS_DIR, output_base_name)
    queue_1_to_2 = multiprocessing.Queue(); queue_2_to_3 = multiprocessing.Queue()
    log_message(main_tag, f"--- Starting App at {time.strftime('%Y/%m/%d %H:%M:%S')} ---", "INFO")
    log_message(main_tag, f"--- Target Domain: {TARGET_DOMAIN_AI} ---", "INFO")
    log_message(main_tag, f"--- Target Description: {TARGET_BUSINESS_DESC} ---", "INFO")
    log_message(main_tag, f"--- Google AI Model: {GOOGLE_AI_MODEL} ---", "INFO")
    log_message(main_tag, f"--- Output File: {OUTPUT_EXCEL_FILE} ---", "INFO")
    try:
        p1 = multiprocessing.Process(target=process_1_crawler, args=(queue_1_to_2, KEYWORDS, ACRONYM, BLACKLIST, WHITELIST, API_KEY_URL_SCAN), name="Crawler")
        p2 = multiprocessing.Process(target=process_2_ai_analyzer, args=(queue_1_to_2, queue_2_to_3, TARGET_DOMAIN_AI, API_KEY_GOOGLE_AI, GOOGLE_AI_MODEL), name="AI-Analyzer-Scorer")
        p3 = multiprocessing.Process(target=process_3_exporter, args=(queue_2_to_3, OUTPUT_EXCEL_FILE), name="Exporter") # P3 không cần target nữa
        log_message(main_tag,"--- Starting Processes ---", "INFO"); p1.start(); p2.start(); p3.start()
        log_message(main_tag,"--- Waiting for Processes ---", "WAIT"); p1.join(); log_message(main_tag, f"P1 Crawler finished.", "INFO"); p2.join(); log_message(main_tag, f"P2 Analyzer/Scorer finished.", "INFO"); p3.join(); log_message(main_tag, f"P3 Exporter finished.", "INFO")
        log_message(main_tag,"\n"+"="*44, "SUCCESS"); log_message(main_tag,"=== App Finished Successfully ===", "SUCCESS"); log_message(main_tag,"="*44, "SUCCESS")
    except KeyboardInterrupt:
        log_message(main_tag, "\n--- Ctrl+C Detected! Terminating... ---", "WARN")
        if 'p1' in locals() and p1.is_alive(): p1.terminate(); log_message(main_tag, "Terminated P1.", "INFO")
        if 'p2' in locals() and p2.is_alive(): p2.terminate(); log_message(main_tag, "Terminated P2.", "INFO")
        if 'p3' in locals() and p3.is_alive(): p3.terminate(); log_message(main_tag, "Terminated P3.", "INFO")
        log_message(main_tag, "--- Processes terminated attempt complete. ---", "INFO")
    except Exception as main_e:
        log_message(main_tag, f"Main execution error: {main_e}", "ERROR"); log_message(main_tag,"Terminating processes...", "WARN")
        if 'p1' in locals() and p1.is_alive(): p1.terminate()
        if 'p2' in locals() and p2.is_alive(): p2.terminate()
        if 'p3' in locals() and p3.is_alive(): p3.terminate()
        log_message(main_tag,"Processes terminated attempt complete.", "INFO")
    finally: colorama.deinit()
