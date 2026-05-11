"""
Chrome Session Manager - Quáº£n lÃ½ Chrome cho proxy server.

TÃ¡i sá»­ dá»¥ng flow tá»« tool (drission_flow_api.py):
1. Má»Ÿ ChromePortable
2. VÃ o labs.google/fx/tools/flow
3. Táº¡o project má»›i
4. Sáºµn sÃ ng nháº­n request

Má»—i request:
1. Inject interceptor (thay bearer token + projectId)
2. Paste prompt â†’ Enter
3. Chá» response â†’ tráº£ káº¿t quáº£
4. Cleanup browser data â†’ sáºµn sÃ ng cho request tiáº¿p
"""
import sys
import os
import time
import json
import base64
import ctypes
import re
import threading
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TOOL_DIR = Path(__file__).parent.parent

# ============================================================
# Flow URL + Constants
# ============================================================
FLOW_URL = "https://labs.google/fx/vi/tools/flow"

# Model name â†’ index trong dropdown
MODEL_INDEX_MAP = {
    'GEM_PIX_2': 0,       # Nano Banana Pro (default)
    'NARWHAL': 1,          # Nano Banana 2
    'IMAGEN_3_5': 2,       # Imagen 4
}

# Clipboard/focus la tai nguyen dung chung cua OS.
# Serialize thao tac paste prompt de tranh worker nay dam worker kia.
_PROMPT_PASTE_LOCK = threading.Lock()

# ============================================================
# JavaScript snippets (tá»« drission_flow_api.py + test_local_proxy.py)
# ============================================================

JS_CLEANUP = """
(function() {
    try { localStorage.clear(); } catch(e) {}
    try { sessionStorage.clear(); } catch(e) {}
    try {
        indexedDB.databases().then(function(dbs) {
            dbs.forEach(function(db) { indexedDB.deleteDatabase(db.name); });
        });
    } catch(e) {}
    try {
        document.cookie.split(";").forEach(function(c) {
            document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
        });
    } catch(e) {}
    try {
        caches.keys().then(function(names) {
            names.forEach(function(name) { caches.delete(name); });
        });
    } catch(e) {}
    try {
        navigator.serviceWorker.getRegistrations().then(function(regs) {
            regs.forEach(function(r) { r.unregister(); });
        });
    } catch(e) {}
    return 'CLEANED';
})();
"""

JS_BLOCK_NEW_TAB = """
(function() {
    try {
        // Chan window.open() de khong mo tab moi.
        window.open = function() { return null; };
    } catch (e) {}

    try {
        // Ep cac link target=_blank ve _self.
        var fix = function(root) {
            var scope = root || document;
            var links = scope.querySelectorAll ? scope.querySelectorAll('a[target="_blank"]') : [];
            for (var i = 0; i < links.length; i++) {
                links[i].setAttribute('target', '_self');
                links[i].removeAttribute('rel');
            }
        };
        fix(document);
        document.addEventListener('click', function(ev) {
            var el = ev.target;
            var a = el && el.closest ? el.closest('a[target="_blank"]') : null;
            if (a) {
                a.setAttribute('target', '_self');
                a.removeAttribute('rel');
            }
        }, true);
        document.addEventListener('DOMContentLoaded', function() { fix(document); }, true);
    } catch (e) {}

    return 'TAB_GUARD_OK';
})();
"""

# v1.0.512: Import tu fingerprint_data.py (pool 65 GPU, 15 screen, 53M+ combos)
from modules.fingerprint_data import (
    FAKE_GPUS as _FAKE_GPUS,
    FAKE_SCREENS as _FAKE_SCREENS,
    FAKE_CORES as _FAKE_CORES,
    FAKE_MEMORY as _FAKE_MEMORY,
    build_fingerprint_js as _build_fingerprint_js,
    get_unique_seed as _get_unique_seed,
)


JS_CLICK_NEW_PROJECT = """
(function() {
    // TÃ¬m nÃºt "Dá»± Ã¡n má»›i" (cÃ³ icon add_2)
    var btns = document.querySelectorAll('button');
    for (var i = 0; i < btns.length; i++) {
        var t = btns[i].textContent.trim();
        if (t.indexOf('add_2') >= 0 || t.indexOf('Dá»± Ã¡n má»›i') >= 0 || t.indexOf('New project') >= 0) {
            btns[i].click();
            return 'CLICKED';
        }
    }

    // Fallback: "Create with Flow" hoáº·c tÆ°Æ¡ng tá»±
    var links = document.querySelectorAll('a');
    for (var i = 0; i < links.length; i++) {
        var t = links[i].textContent.trim();
        if (t.indexOf('Create') >= 0 && t.indexOf('Flow') >= 0) {
            links[i].click();
            return 'CLICKED_LINK';
        }
    }

    return 'NOT_FOUND';
})();
"""

JS_SELECT_MODEL = """
(function(modelIndex) {
    window._modelSelectResult = 'PENDING';
    var keywords = ['Banana', 'Imagen', 'Veo', 'Video', 'Fast'];
    var btns = document.querySelectorAll('button');
    var halfH = window.innerHeight * 0.5;
    var btn1 = null;
    for (var i = 0; i < btns.length; i++) {
        var t = btns[i].textContent.trim();
        var rect = btns[i].getBoundingClientRect();
        if (rect.width > 50 && rect.y > halfH) {
            for (var k = 0; k < keywords.length; k++) {
                if (t.indexOf(keywords[k]) >= 0) { btn1 = btns[i]; break; }
            }
            if (btn1) break;
        }
    }
    if (btn1) {
        btn1.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
        btn1.dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
    } else { window._modelSelectResult = 'NO_BUTTON'; return; }

    setTimeout(function() {
        var tab = document.querySelector('[id*="trigger-IMAGE"]');
        if (tab) {
            tab.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
            tab.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
            tab.dispatchEvent(new MouseEvent('click', {bubbles: true}));
        }
        setTimeout(function() {
            var t169 = document.querySelector('[id*="trigger-LANDSCAPE"]');
            if (t169) {
                t169.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
                t169.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
                t169.dispatchEvent(new MouseEvent('click', {bubbles: true}));
            }
            setTimeout(function() {
                var x1 = document.querySelector('[id*="trigger-1"]');
                if (x1) {
                    x1.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
                    x1.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
                    x1.dispatchEvent(new MouseEvent('click', {bubbles: true}));
                }
                setTimeout(function() {
                    var btns = document.querySelectorAll('button');
                    for (var i = 0; i < btns.length; i++) {
                        var t = btns[i].textContent.trim();
                        if (t.indexOf('arrow_drop_down') >= 0 && btns[i].getBoundingClientRect().width > 0) {
                            btns[i].dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
                            btns[i].dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
                            break;
                        }
                    }
                    setTimeout(function() {
                        var items = document.querySelectorAll('[role="menuitem"]');
                        if (items.length > modelIndex) {
                            items[modelIndex].dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
                            items[modelIndex].dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
                            items[modelIndex].click();
                            window._modelSelectResult = 'SELECTED_' + modelIndex;
                        } else { window._modelSelectResult = 'NO_ITEMS'; }
                        setTimeout(function() {
                            document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
                            document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
                        }, 500);
                    }, 800);
                }, 500);
            }, 300);
        }, 300);
    }, 1500);
})(MODEL_INDEX);
"""

# v1.0.634: JS chuyen sang Video mode
# Flow: Open settings â†’ Video tab â†’ Thanh phan (VIDEO_REFERENCES) â†’ LANDSCAPE â†’ x1 â†’ chon model bat ky
# Khong bat buoc Lower Priority - server Chrome chi can o Video mode, interceptor se thay model tu VM
JS_SELECT_VIDEO_MODE = """
(function() {
    window._videoModeResult = 'PENDING';

    // 1. Check panel da mo chua, neu chua thi mo settings
    var vidTab = document.querySelector('[id*="trigger-IMAGE"]');
    if (!vidTab) vidTab = document.querySelector('[id*="trigger-VIDEO"]:not([id*="FRAMES"]):not([id*="REFERENCES"])');
    if (!vidTab || vidTab.getBoundingClientRect().width === 0) {
        var keywords = ['Banana', 'Imagen', 'Veo', 'Video', 'Fast'];
        var btns = document.querySelectorAll('button');
        var halfH = window.innerHeight * 0.5;
        var btn1 = null;
        for (var i = 0; i < btns.length; i++) {
            var t = btns[i].textContent.trim();
            var rect = btns[i].getBoundingClientRect();
            if (rect.width > 50 && rect.y > halfH) {
                for (var k = 0; k < keywords.length; k++) {
                    if (t.indexOf(keywords[k]) >= 0) { btn1 = btns[i]; break; }
                }
                if (btn1) break;
            }
        }
        if (btn1) {
            btn1.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
            btn1.dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
            console.log('[VIDEO-MODE] 1. Opened settings: ' + btn1.textContent.trim().substring(0, 40));
        } else {
            console.log('[VIDEO-MODE] 1. WARN: No settings button found');
        }
    } else {
        console.log('[VIDEO-MODE] 1. Panel already open');
    }

    setTimeout(function() {
        // 2. Click tab Video (MouseEvent cho Radix)
        var tabs = document.querySelectorAll('[role="tab"]');
        for (var i = 0; i < tabs.length; i++) {
            if ((tabs[i].id || '').indexOf('trigger-VIDEO') >= 0 && (tabs[i].id || '').indexOf('FRAMES') < 0 && (tabs[i].id || '').indexOf('REFERENCES') < 0) {
                tabs[i].dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
                tabs[i].dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
                tabs[i].dispatchEvent(new MouseEvent('click', {bubbles: true}));
                console.log('[VIDEO-MODE] 2. Video tab: ' + tabs[i].getAttribute('data-state'));
                break;
            }
        }

        // 3. Click Thanh phan (VIDEO_REFERENCES) = I2V mode
        setTimeout(function() {
            var tp = document.querySelector('[id*="trigger-VIDEO_REFERENCES"]');
            if (tp) {
                tp.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
                tp.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
                tp.dispatchEvent(new MouseEvent('click', {bubbles: true}));
                console.log('[VIDEO-MODE] 3. Thanh phan: ' + tp.getAttribute('data-state'));
            }

            setTimeout(function() {
                // 4. Click LANDSCAPE (16:9)
                var t169 = document.querySelector('[id*="trigger-LANDSCAPE"]');
                if (t169) {
                    t169.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
                    t169.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
                    t169.dispatchEvent(new MouseEvent('click', {bubbles: true}));
                    console.log('[VIDEO-MODE] 4. 16:9');
                }

                // 5. Click x1
                setTimeout(function() {
                    var x1 = document.querySelector('[id*="trigger-1"]');
                    if (x1) {
                        x1.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
                        x1.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
                        x1.dispatchEvent(new MouseEvent('click', {bubbles: true}));
                        console.log('[VIDEO-MODE] 5. x1');
                    }

                    // 6. Mo model dropdown
                    setTimeout(function() {
                        var btns = document.querySelectorAll('button');
                        for (var i = 0; i < btns.length; i++) {
                            var t = btns[i].textContent.trim();
                            if (t.indexOf('arrow_drop_down') >= 0 && (t.indexOf('Veo') >= 0 || t.indexOf('Fast') >= 0)) {
                                btns[i].dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
                                btns[i].dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
                                console.log('[VIDEO-MODE] 6. Dropdown: ' + t.substring(0, 40));
                                break;
                            }
                        }

                        // 7. Chon Veo 3.1 - Lite [Lower Priority]
                        setTimeout(function() {
                            var items = document.querySelectorAll('[role="menuitem"]');
                            var target = null;
                            var fallback = null;
                            for (var i = 0; i < items.length; i++) {
                                var label = (items[i].textContent || '').replace(/\s+/g, ' ').trim();
                                if (label.indexOf('Lower Priority') >= 0) {
                                    if (!fallback) fallback = items[i];
                                    if (label.indexOf('Lite') >= 0) { target = items[i]; break; }
                                }
                            }
                            target = target || fallback;
                            if (target) {
                                target.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
                                target.dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
                                target.click();
                                console.log('[VIDEO-MODE] 7. Selected: ' + target.textContent.substring(0, 60));
                                window._videoModeResult = 'SUCCESS';
                            } else {
                                console.log('[VIDEO-MODE] 7. WARN: Veo Lite Lower Priority not found');
                                window._videoModeResult = 'NO_LOWER_PRIORITY_MODEL';
                            }
                            // 8. Dong menu
                            setTimeout(function() {
                                document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
                                document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
                                console.log('[VIDEO-MODE] 8. Done! ' + window._videoModeResult);
                            }, 500);
                        }, 800);
                    }, 500);
                }, 300);
            }, 300);
        }, 500);
    }, 1500);
})();
"""


def build_video_interceptor_js(client_bearer_token: str, client_project_id: str,
                                media_id: str, video_model: str = '') -> str:
    """
    v1.0.687: Video interceptor - THAY bearer token + projectId + inject mediaId.
    KHONG override videoModelKey - de Chrome UI quyet dinh model (Lite LP).

    Chrome DA O VIDEO MODE (da click chuyen sang video UI).
    Interceptor:
    - Thay token -> client's token (VM co quyen Lower Priority)
    - Thay projectId -> client's projectId
    - Inject referenceImages voi mediaId (anh lam khung dau)
    - Convert T2V -> I2V model (giu nguyen model Chrome UI da chon)
    - Chi override videoModelKey khi client gui model cu the (video_model != '')
    - GIU NGUYEN recaptchaToken tu Chrome (hop le cho video mode)
    """
    safe_token = client_bearer_token.replace("\\", "\\\\").replace("'", "\\'")
    safe_project = client_project_id.replace("\\", "\\\\").replace("'", "\\'")
    safe_media = media_id.replace("\\", "\\\\").replace("'", "\\'")
    safe_model = video_model.replace("\\", "\\\\").replace("'", "\\'") if video_model else ''

    return """
window._response = null;
window._responseError = null;
window._requestPending = false;
window._videoResponse = null;
window._videoError = null;
window._videoPending = false;
window._clientBearerToken = '""" + safe_token + """';
window._clientProjectId = '""" + safe_project + """';
window._clientMediaId = '""" + safe_media + """';
window._clientVideoModel = '""" + safe_model + """';

return (function() {
    if (window.__proxyInterceptReady) return 'ALREADY';
    window.__proxyInterceptReady = true;

    var origFetch = window.fetch;
    window.fetch = async function(url, opts) {
        var urlStr = typeof url === 'string' ? url : url.url;

        // Catch VIDEO request (batchAsync... hoac batchGenerate)
        if (urlStr.includes('aisandbox') && (urlStr.includes('batchAsync') || urlStr.includes('batchGenerate') || urlStr.includes('flowMedia') || urlStr.includes('video:'))) {
            console.log('[VIDEO-PROXY] Caught request:', urlStr.substring(0, 100));

            window._videoPending = true;
            window._videoResponse = null;
            window._videoError = null;

            // 1. THAY BEARER TOKEN
            if (window._clientBearerToken && opts) {
                if (!opts.headers) opts.headers = {};
                if (opts.headers instanceof Headers) {
                    opts.headers.set('Authorization', 'Bearer ' + window._clientBearerToken);
                } else {
                    opts.headers['Authorization'] = 'Bearer ' + window._clientBearerToken;
                }
                console.log('[VIDEO-PROXY] 1. Replaced bearer token');
            }

            // v1.0.636: DOI URL T2V â†’ I2V (giong API mode)
            // Chrome gui batchAsyncGenerateVideoText (T2V) nhung ta can batchAsyncGenerateVideoReferenceImages (I2V)
            // Endpoint T2V KHONG chap nhan field referenceImages â†’ 400 error
            var actualUrl = url;
            if (urlStr.includes('batchAsyncGenerateVideoText')) {
                var newUrlStr = urlStr.replace('batchAsyncGenerateVideoText', 'batchAsyncGenerateVideoReferenceImages');
                actualUrl = newUrlStr;
                console.log('[VIDEO-PROXY] 1b. URL T2V â†’ I2V:', newUrlStr.substring(newUrlStr.length - 60));
            }

            // 2. THAY projectId + inject mediaId trong body
            if (opts && opts.body) {
                try {
                    var body = JSON.parse(opts.body);

                    // Thay projectId
                    if (body.clientContext && window._clientProjectId) {
                        body.clientContext.projectId = window._clientProjectId;
                        console.log('[VIDEO-PROXY] 2a. ProjectId â†’ ' + window._clientProjectId);
                    }

                    // Inject referenceImages voi mediaId + thay videoModelKey
                    if (body.requests) {
                        body.requests.forEach(function(req) {
                            if (window._clientMediaId) {
                                req.referenceImages = [{
                                    imageUsageType: 'IMAGE_USAGE_TYPE_ASSET',
                                    mediaId: window._clientMediaId
                                }];
                                console.log('[VIDEO-PROXY] 2b. Injected mediaId: ' + window._clientMediaId.substring(0, 50) + '...');
                            }
                            // v1.0.636: Doi model T2V â†’ I2V (giong API mode)
                            // Chrome gui model t2v, can doi thanh r2v cho I2V endpoint
                            var currentModel = req.videoModelKey || '';
                            if (currentModel.includes('_t2v_')) {
                                var newModel = currentModel.replace('_t2v_', '_r2v_');
                                // I2V model can _landscape truoc _ultra
                                if (newModel.includes('_ultra') && !newModel.includes('_landscape')) {
                                    newModel = newModel.replace('_ultra', '_landscape_ultra');
                                }
                                req.videoModelKey = newModel;
                                console.log('[VIDEO-PROXY] 2c-model. T2Vâ†’I2V model: ' + currentModel + ' â†’ ' + newModel);
                            }
                            // Override neu VM chi dinh model cu the
                            if (window._clientVideoModel) {
                                req.videoModelKey = window._clientVideoModel;
                                console.log('[VIDEO-PROXY] 2c-override. videoModelKey â†’ ' + window._clientVideoModel);
                            }
                        });
                    }

                    // GIU NGUYEN recaptchaToken tu Chrome!
                    var recaptcha = body.clientContext ? body.clientContext.recaptchaToken : '';
                    console.log('[VIDEO-PROXY] 2c. recaptchaToken kept: ' + (recaptcha ? recaptcha.substring(0, 20) + '...' : 'EMPTY'));

                    opts.body = JSON.stringify(body);
                } catch(e) {
                    console.log('[VIDEO-PROXY] Parse body error:', e);
                }
            }

            try {
                var response = await origFetch.apply(this, [actualUrl, opts]);
                var cloned = response.clone();
                try {
                    var data = await cloned.json();
                    console.log('[VIDEO-PROXY] Response status:', response.status);

                    if (response.status === 403 || (data.error && data.error.code === 403)) {
                        window._videoResponse = {error: data.error || {code: 403, message: 'Permission denied'}};
                        window._videoError = 'Error 403';
                    } else if (response.status === 429 || (data.error && (data.error.code === 429 || data.error.code === 253))) {
                        window._videoResponse = {error: data.error || {code: 429, message: 'Quota exceeded'}};
                        window._videoError = 'Error 429';
                    } else if (data.operations) {
                        console.log('[VIDEO-PROXY] Got ' + data.operations.length + ' operations!');
                        window._videoResponse = data;
                    } else if (data.error) {
                        window._videoResponse = {error: data.error};
                        window._videoError = 'Error: ' + (data.error.message || JSON.stringify(data.error));
                    } else {
                        window._videoResponse = data;
                    }
                } catch(e) {
                    window._videoResponse = {status: response.status, error: 'parse_failed'};
                }
                window._videoPending = false;
                return response;
            } catch(e) {
                window._videoError = 'Fetch error: ' + e.message;
                window._videoPending = false;
                return origFetch.apply(this, [url, opts]);
            }
        }

        return origFetch.apply(this, [url, opts]);
    };

    return 'PROXY_INTERCEPTOR_READY';
})();
"""


def build_interceptor_js(client_bearer_token: str, client_project_id: str,
                         image_inputs: list = None) -> str:
    """
    JS Interceptor - THAY bearer token + projectId + inject imageInputs.

    Flow:
    1. Chrome gá»­i request vá»›i recaptchaToken há»£p lá»‡
    2. Interceptor THAY: bearer token â†’ client's token, projectId â†’ client's projectId
    3. Inject imageInputs (reference images) náº¿u cÃ³
    4. GIá»® NGUYÃŠN recaptchaToken tá»« Chrome
    5. Google nháº­n: token khÃ¡ch + captcha há»£p lá»‡ â†’ táº¡o áº£nh dÆ°á»›i tÃ i khoáº£n khÃ¡ch
    """
    safe_token = client_bearer_token.replace("\\", "\\\\").replace("'", "\\'")
    safe_project = client_project_id.replace("\\", "\\\\").replace("'", "\\'")

    # imageInputs JSON (hoáº·c null náº¿u khÃ´ng cÃ³)
    if image_inputs and len(image_inputs) > 0:
        import json
        image_inputs_json = json.dumps(image_inputs)
    else:
        image_inputs_json = "null"

    # v1.0.548: Dung return NGAY DAU de DrissionPage bat duoc gia tri tra ve
    return """
window._response = null;
window._responseError = null;
window._requestPending = false;
window._clientBearerToken = '""" + safe_token + """';
window._clientProjectId = '""" + safe_project + """';
window._imageInputs = """ + image_inputs_json + """;

return (function() {
    if (window.__proxyInterceptReady) return 'ALREADY';
    window.__proxyInterceptReady = true;

    var origFetch = window.fetch;
    window.fetch = async function(url, opts) {
        var urlStr = typeof url === 'string' ? url : url.url;

        if (urlStr.includes('aisandbox') && (urlStr.includes('batchGenerate') || urlStr.includes('flowMedia'))) {
            console.log('[PROXY] Caught request:', urlStr);

            if (!window._response) {
                window._requestPending = true;
                window._response = null;
                window._responseError = null;
            }

            // 1. THAY BEARER TOKEN
            if (window._clientBearerToken && opts) {
                if (!opts.headers) opts.headers = {};
                if (opts.headers instanceof Headers) {
                    opts.headers.set('Authorization', 'Bearer ' + window._clientBearerToken);
                } else {
                    opts.headers['Authorization'] = 'Bearer ' + window._clientBearerToken;
                }
                console.log('[PROXY] 1. Replaced bearer token');
            }

            // 2. THAY projectId + inject imageInputs trong body
            // QUAN TRONG: Chi xu ly khi co clientProjectId (giong test file)
            if (window._clientProjectId && opts && opts.body) {
                try {
                    var body = JSON.parse(opts.body);

                    // Thay projectId trong clientContext
                    if (body.clientContext) {
                        var oldProjId = body.clientContext.projectId;
                        body.clientContext.projectId = window._clientProjectId;
                        console.log('[PROXY] 2a. Body projectId: ' + oldProjId + ' â†’ ' + window._clientProjectId);
                    }

                    // Thay projectId trong má»—i request
                    if (body.requests) {
                        body.requests.forEach(function(req) {
                            if (req.clientContext) {
                                req.clientContext.projectId = window._clientProjectId;
                            }
                            console.log('[PROXY] 2b. Prompt: ' + (req.prompt || '').substring(0, 60));
                        });
                    }

                    // Inject imageInputs (reference images) náº¿u cÃ³
                    if (window._imageInputs && body.requests) {
                        body.requests.forEach(function(req) {
                            req.imageInputs = window._imageInputs;
                        });
                        console.log('[PROXY] 2c. Injected ' + window._imageInputs.length + ' reference images');
                    }

                    // GIá»® NGUYÃŠN recaptchaToken tá»« Chrome!
                    var recaptcha = body.clientContext ? body.clientContext.recaptchaToken : '';
                    console.log('[PROXY] 2d. recaptchaToken kept: ' + (recaptcha ? recaptcha.substring(0, 20) + '...' : 'EMPTY'));

                    opts.body = JSON.stringify(body);
                } catch(e) {
                    console.log('[PROXY] Parse body error:', e);
                }
            }

            // 3. THAY projectId trong URL
            if (window._clientProjectId) {
                var newUrl = urlStr.replace(/projects\\/[^/]+/, 'projects/' + window._clientProjectId);
                if (newUrl !== urlStr) {
                    console.log('[PROXY] 3. URL project replaced');
                    url = newUrl;
                }
            }

            try {
                var response = await origFetch.apply(this, [url, opts]);
                var cloned = response.clone();
                var data = await cloned.json();

                console.log('[PROXY] Status:', response.status);

                if (response.status === 200 && data.media) {
                    window._response = data;
                    console.log('[PROXY] OK! ' + data.media.length + ' images');
                } else if (data.error) {
                    window._response = {error: data.error};
                    window._responseError = 'Error ' + (data.error.code || response.status) + ': ' + (data.error.message || '');
                    console.log('[PROXY] ERROR:', window._responseError);
                } else {
                    window._response = data;
                }

                window._requestPending = false;
                return response;
            } catch(e) {
                window._responseError = 'Fetch error: ' + e.message;
                window._requestPending = false;
                return origFetch.apply(this, [url, opts]);
            }
        }

        return origFetch.apply(this, [url, opts]);
    };

    return 'PROXY_INTERCEPTOR_READY';
})();
"""


def get_server_accounts() -> list:
    """
    Äá»c danh sÃ¡ch tÃ i khoáº£n server tá»« Google Sheet.
    Sheet: "SERVER", Cá»™t B: tÃ i khoáº£n (format: email|password|2fa má»—i dÃ²ng)

    Returns: [{"id": "email", "password": "pass", "totp_secret": "..."}, ...]
    """
    try:
        sys.path.insert(0, str(TOOL_DIR))
        from google_login import load_gsheet_client, parse_accounts_cell, col_letter_to_index

        gc, spreadsheet_name = load_gsheet_client()
        if not gc:
            print("[SERVER] Khong load duoc Google Sheet client")
            return []

        ws = gc.open(spreadsheet_name).worksheet("SERVER")
        all_data = ws.get_all_values()

        if not all_data:
            print("[SERVER] Sheet 'SERVER' trong!")
            return []

        # Cot B (index 1) chua tai khoan - doc tat ca dong
        accounts = []
        col_b = col_letter_to_index("B")  # = 1

        for row_idx, row in enumerate(all_data, start=1):
            if len(row) <= col_b:
                continue

            cell_value = str(row[col_b]).strip()
            if not cell_value:
                continue

            # Parse tung dong trong cell (co the co nhieu tai khoan trong 1 cell)
            parsed = parse_accounts_cell(cell_value)
            if parsed:
                accounts.extend(parsed)

        if accounts:
            print(f"[SERVER] Tim thay {len(accounts)} tai khoan tu sheet 'SERVER':")
            for i, acc in enumerate(accounts):
                has_2fa = " [2FA]" if acc.get('totp_secret') else ""
                print(f"  {i+1}. {acc['id']}{has_2fa}")

        return accounts

    except Exception as e:
        print(f"[SERVER] Loi doc sheet 'SERVER': {e}")
        return []


class ChromeSession:
    """Quáº£n lÃ½ Chrome session cho proxy server."""

    def __init__(self, chrome_portable_path: str = None, port: int = 19222, ipv6: str = "",
                 proxy_provider=None):
        """
        Args:
            chrome_portable_path: ÄÆ°á»ng dáº«n tá»›i ChromePortable.exe
                                  Máº·c Ä‘á»‹nh: {tool_dir}/GoogleChromePortable/GoogleChromePortable.exe
            port: Debug port cho Chrome (máº·c Ä‘á»‹nh 19222 - khÃ´ng trÃ¹ng vá»›i workers 9222/9223)
            ipv6: IPv6 address - náº¿u cÃ³ sáº½ táº¡o SOCKS5 proxy Ä‘á»ƒ Chrome dÃ¹ng IPv6
            proxy_provider: ProxyProvider instance (v1.0.545) - neu co thi dung thay IPv6
        """
        if chrome_portable_path:
            self.chrome_path = Path(chrome_portable_path)
        else:
            self.chrome_path = TOOL_DIR / "GoogleChromePortable" / "GoogleChromePortable.exe"

        self.chrome_data = self.chrome_path.parent / "Data" / "profile"
        self.port = port
        self.ipv6 = ipv6.strip() if ipv6 else ""
        self._proxy_port = 0  # SOCKS5 proxy port (set khi start proxy)
        self._proxy = None  # IPv6SocksProxy instance
        # v1.0.545: ProxyProvider interface (thay the IPv6 truc tiep)
        self._proxy_provider = proxy_provider
        self.page = None
        self.ready = False
        self.project_url = None
        self._image_mode_selected = False
        self._account = None  # Tai khoan dang dung
        # v1.0.512: LUON tao fingerprint tu dau, khong doi 403
        self._fingerprint_seed = _get_unique_seed()
        self._consecutive_403 = 0   # Dem 403 lien tiep
        self._current_model_index = 0  # 0=Nano Banana Pro, 1=Nano Banana 2, 2=Imagen 4
        self._cleared_data_for_403 = False  # Da clear data chua

    @staticmethod
    def _fix_mojibake_text(msg):
        """Co gang sua chuoi bi loi ma hoa (vd: 'Táº¡o' -> 'Tạo')."""
        if not isinstance(msg, str) or not msg:
            return msg

        def _bad_score(s: str) -> int:
            markers = ["Ã", "Ä", "Æ", "Å", "â†", "á»", "�"]
            return sum(s.count(m) for m in markers)

        best = msg
        best_score = _bad_score(msg)
        for enc in ("latin1", "cp1252"):
            try:
                fixed = msg.encode(enc).decode("utf-8")
            except Exception:
                continue
            score = _bad_score(fixed)
            if score < best_score:
                best = fixed
                best_score = score
        return best

    def log(self, msg: str, level: str = "INFO"):
        prefix = {"INFO": "[INFO]", "OK": "[OK]", "WARN": "[WARN]", "ERROR": "[ERROR]"}
        msg = self._fix_mojibake_text(msg)
        print(f"  {prefix.get(level, '[INFO]')} {msg}")

    def _get_window_slot(self) -> int:
        """
        Xac dinh slot cua Chrome theo ten folder portable.
        Tra ve 0..N (fallback theo port neu khong match ten folder).
        """
        try:
            folder = self.chrome_path.parent.name
            if folder == "GoogleChromePortable":
                return 0
            if folder == "GoogleChromePortable - Copy":
                return 1
            m = re.match(r"GoogleChromePortable - Copy \((\d+)\)$", folder)
            if m:
                return max(0, int(m.group(1)))
            m2 = re.match(r"GoogleChromePortable_(\d+)$", folder)
            if m2:
                return max(0, int(m2.group(1)) - 1)
        except Exception:
            pass

        # Fallback theo port server worker: 19222 -> 0, 19223 -> 1, ...
        return max(0, int(self.port) - 19222)

    def _calc_window_layout(self):
        """Tinh layout cua so theo man hinh hien tai."""
        slot = self._get_window_slot()
        sw = int(ctypes.windll.user32.GetSystemMetrics(0))
        sh = int(ctypes.windll.user32.GetSystemMetrics(1))

        total_slots = max(1, int(os.getenv("CHROME_LAYOUT_SLOTS", "10")))
        # 4K: 3 cot. QHD (2560): 2 cot x 5 hang cho 10 Chrome.
        if sw >= 3800:
            default_cols = "3"
        else:
            default_cols = "2"
        cols = max(1, int(os.getenv("CHROME_LAYOUT_COLS", default_cols)))
        rows = max(1, (total_slots + cols - 1) // cols)

        # De vung ben trai cho tool (co the override bang env).
        # 4K: chua nua man hinh trai. QHD: chua ~960px ben trai cho GUI.
        if sw >= 3800:
            default_reserve = "1920"
        elif sw >= 2500:
            default_reserve = "960"
        else:
            default_reserve = "760"
        reserve_left = int(os.getenv("CHROME_LAYOUT_LEFT_RESERVED", default_reserve))
        reserve_left = max(0, min(reserve_left, sw - 300))

        usable_w = max(300, sw - reserve_left)
        cell_w = max(320, usable_w // cols)
        cell_h = max(180, sh // rows)

        slot = max(0, min(slot, total_slots - 1))
        col = slot % cols
        row = slot // cols

        x = reserve_left + col * cell_w
        y = row * cell_h
        return slot, x, y, cell_w, cell_h

    def _apply_window_layout_options(self, co):
        """
        Co dinh vi tri Chrome theo luoi de de quan sat cung GUI.
        - Chu truong: de lai 1 vung ben trai cho tool.
        - Mac dinh: 10 Chrome -> 2 cot x 5 hang.
        """
        try:
            slot, x, y, cell_w, cell_h = self._calc_window_layout()

            # Truoc khi set argument moi -> clear argument cu bang cach ghi de them gia tri moi
            co.set_argument(f'--window-position={x},{y}')
            co.set_argument(f'--window-size={cell_w},{cell_h}')
            self.log(f"[LAYOUT] slot={slot} pos=({x},{y}) size=({cell_w}x{cell_h})")
        except Exception as e:
            self.log(f"[LAYOUT] skip: {e}", "WARN")

    def _enforce_window_layout(self):
        """Ep lai vi tri/size bang CDP sau khi Chrome da mo."""
        try:
            if not self.page:
                return
            slot, x, y, cell_w, cell_h = self._calc_window_layout()
            info = self.page.run_cdp('Browser.getWindowForTarget')
            window_id = info.get('windowId')
            if not window_id:
                self.log("[LAYOUT] CDP: khong lay duoc windowId", "WARN")
                return
            self.page.run_cdp(
                'Browser.setWindowBounds',
                windowId=window_id,
                bounds={
                    'left': int(x),
                    'top': int(y),
                    'width': int(cell_w),
                    'height': int(cell_h),
                    'windowState': 'normal',
                }
            )
            self.log(f"[LAYOUT-CDP] slot={slot} pos=({x},{y}) size=({cell_w}x{cell_h})")
        except Exception as e:
            self.log(f"[LAYOUT-CDP] skip: {e}", "WARN")

    # ============================================================
    # Fingerprint Spoof - Doi fingerprint khi bi 403
    # ============================================================

    def inject_fingerprint_spoof(self):
        """
        Inject fingerprint spoof JS.

        Su dung CDP Page.addScriptToEvaluateOnNewDocument de inject TRUOC khi
        page scripts chay â†’ Google khong thay fingerprint that.
        Dong thoi chay run_js() cho page hien tai.
        """
        if self._fingerprint_seed <= 0 or not self.page:
            return
        try:
            js = _build_fingerprint_js(self._fingerprint_seed)

            # 1. CDP: Inject cho TAT CA page loads sau nay (TRUOC khi scripts chay)
            try:
                # Xoa script cu (neu co)
                if hasattr(self, '_spoof_script_id') and self._spoof_script_id:
                    try:
                        self.page.run_cdp('Page.removeScriptToEvaluateOnNewDocument',
                                          identifier=self._spoof_script_id)
                    except Exception:
                        pass

                result = self.page.run_cdp('Page.addScriptToEvaluateOnNewDocument',
                                            source=js)
                self._spoof_script_id = result.get('identifier', '')
                self.log(f"[SPOOF] CDP pre-load inject OK (seed={self._fingerprint_seed})")
            except Exception as e:
                self.log(f"[SPOOF] CDP inject failed ({e}), fallback run_js", "WARN")

            # 2. run_js: Cho page HIEN TAI (da load roi)
            self.page.run_js(js)

            # 3. VERIFY: Kiem tra fingerprint co hoat dong khong
            try:
                verify = self.page.run_js("""
                    var c = navigator.hardwareConcurrency;
                    var m = navigator.deviceMemory || 'N/A';
                    var w = screen.width;
                    var h = screen.height;
                    var gl = null;
                    try {
                        var canvas = document.createElement('canvas');
                        var ctx = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                        if (ctx) gl = ctx.getParameter(37446);
                    } catch(e) {}
                    return 'cores=' + c + ' mem=' + m + ' screen=' + w + 'x' + h + ' gpu=' + (gl || 'N/A').substring(0, 40);
                """)
                self.log(f"[SPOOF] VERIFY: {verify}")
            except Exception:
                pass
        except Exception as e:
            self.log(f"[SPOOF] Inject error: {e}", "WARN")

    def apply_page_zoom(self, retries: int = 3, force_register_script: bool = True) -> bool:
        """
        Dat zoom cho trang hien tai.
        Mac dinh 50% de UI nho gon hon tren cua so worker.
        Override bang env CHROME_PAGE_ZOOM (vd: 60, 80, 100).
        """
        if not self.page:
            return False
        try:
            zoom_val = int(os.getenv("CHROME_PAGE_ZOOM", "50"))
            zoom_val = max(25, min(200, zoom_val))
        except Exception:
            zoom_val = 50

        target = f"{zoom_val}%"
        scale = max(0.25, min(2.0, zoom_val / 100.0))

        zoom_reset_js = """
            (function() {
                try {
                    try { document.documentElement.style.zoom = '100%'; } catch(e) {}
                    try { if (document.body) document.body.style.zoom = '100%'; } catch(e) {}
                    return '100%';
                } catch(e) {
                    return 'ERR:' + e;
                }
            })();
        """

        zoom_apply_js = f"""
            (function() {{
                try {{
                    var z = '{target}';
                    try {{ document.documentElement.style.zoom = z; }} catch(e) {{}}
                    try {{ if (document.body) document.body.style.zoom = '100%'; }} catch(e) {{}}
                    return (document.documentElement && document.documentElement.style.zoom) || '';
                }} catch(e) {{
                    return 'ERR:' + e;
                }}
            }})();
        """

        zoom_verify_js = """
            (function() {
                try {
                    var dz = '';
                    try { dz = (document.documentElement && document.documentElement.style.zoom) || ''; } catch(e) {}
                    var vv = '';
                    try {
                        if (window.visualViewport && window.visualViewport.scale != null) {
                            vv = String(window.visualViewport.scale);
                        }
                    } catch(e) {}
                    return JSON.stringify({dz: dz, vv: vv});
                } catch(e) {
                    return JSON.stringify({dz: '', vv: '', err: String(e)});
                }
            })();
        """

        zoom_bootstrap_js = f"""
            (function() {{
                try {{
                    var z = '{target}';
                    var applyZoom = function() {{
                        try {{ document.documentElement.style.zoom = z; }} catch(e) {{}}
                        try {{ if (document.body) document.body.style.zoom = '100%'; }} catch(e) {{}}
                    }};
                    try {{ applyZoom(); }} catch(e) {{}}
                    try {{ document.addEventListener('DOMContentLoaded', applyZoom, true); }} catch(e) {{}}
                    try {{ window.addEventListener('load', applyZoom, true); }} catch(e) {{}}
                }} catch(e) {{}}
            }})();
        """

        try:
            if force_register_script:
                try:
                    if hasattr(self, '_zoom_script_id') and self._zoom_script_id:
                        try:
                            self.page.run_cdp(
                                'Page.removeScriptToEvaluateOnNewDocument',
                                identifier=self._zoom_script_id
                            )
                        except Exception:
                            pass
                    res = self.page.run_cdp(
                        'Page.addScriptToEvaluateOnNewDocument',
                        source=zoom_bootstrap_js
                    )
                    self._zoom_script_id = res.get('identifier', '')
                except Exception as cdp_e:
                    self.log(f"[ZOOM] CDP pre-load inject failed: {cdp_e}", "WARN")

            # Set + verify thuc te.
            cdp_scale_ok = False
            for i in range(max(1, retries)):
                try:
                    self.page.run_js(zoom_reset_js)
                except Exception:
                    pass

                # CDP scale fallback: tranh mat zoom sau reload tren mot so may.
                try:
                    self.page.run_cdp('Emulation.setPageScaleFactor', pageScaleFactor=1.0)
                    self.page.run_cdp('Emulation.setPageScaleFactor', pageScaleFactor=scale)
                    cdp_scale_ok = True
                except Exception:
                    pass

                actual = ''
                vv = ''
                try:
                    self.page.run_js(zoom_apply_js)
                    raw = self.page.run_js(zoom_verify_js)
                    parsed = json.loads(raw) if isinstance(raw, str) and raw else {}
                    actual = str(parsed.get('dz') or '').strip()
                    vv = str(parsed.get('vv') or '').strip()
                except Exception:
                    pass

                if actual == target:
                    self.log(f"[ZOOM] Verified: {actual}")
                    return True

                try:
                    if vv:
                        vv_val = float(vv)
                        if abs(vv_val - scale) <= 0.05:
                            self.log(f"[ZOOM] Verified via viewport scale: {vv_val:.2f}")
                            return True
                except Exception:
                    pass

                # Khong doc duoc style nhung CDP scale da set thanh cong -> coi nhu OK.
                if cdp_scale_ok and not actual:
                    self.log(f"[ZOOM] Applied (verify-unavailable), target={target}")
                    return True
                time.sleep(0.2)

            self.log(f"[ZOOM] MISMATCH target={target}, actual={actual}, vv={vv}", "WARN")
            return False
        except Exception as e:
            self.log(f"[ZOOM] Set zoom failed: {e}", "WARN")
            return False

    def inject_tab_guard(self):
        """
        Chan mo tab moi cho session hien tai.
        CDP de ap dung cho page load tiep theo + run_js cho page dang mo.
        """
        if not self.page:
            return
        try:
            try:
                if hasattr(self, '_tab_guard_script_id') and self._tab_guard_script_id:
                    try:
                        self.page.run_cdp(
                            'Page.removeScriptToEvaluateOnNewDocument',
                            identifier=self._tab_guard_script_id
                        )
                    except Exception:
                        pass

                result = self.page.run_cdp(
                    'Page.addScriptToEvaluateOnNewDocument',
                    source=JS_BLOCK_NEW_TAB
                )
                self._tab_guard_script_id = result.get('identifier', '')
            except Exception as e:
                self.log(f"[TAB] CDP tab-guard inject failed: {e}", "WARN")

            try:
                self.page.run_js(JS_BLOCK_NEW_TAB)
            except Exception as e:
                self.log(f"[TAB] Runtime tab-guard inject failed: {e}", "WARN")
        except Exception as e:
            self.log(f"[TAB] Tab-guard error: {e}", "WARN")

    def rotate_proxy(self, reason: str = "403") -> bool:
        """
        v1.0.545: Doi IP qua ProxyProvider (IPv6/Webshare/...).
        Goi method nay thay vi rotate_ipv6() khi dung ProxyProvider.

        Returns: True neu doi thanh cong.
        """
        if self._proxy_provider:
            ok = self._proxy_provider.rotate(reason)
            if ok:
                self.log(f"[PROXY] Rotated ({reason}): â†’ {self._proxy_provider.get_current_ip()}", "OK")
            else:
                self.log(f"[PROXY] Rotate failed ({reason})", "WARN")
            return ok
        # Fallback: rotate_ipv6
        return self.rotate_ipv6()

    def rotate_ipv6(self, new_ip: str = "") -> bool:
        """
        Doi IPv6 address va restart SOCKS5 proxy.

        Args:
            new_ip: IPv6 moi (tu ChromePool). Neu rong â†’ tu tim tu ipv6_rotator.

        Returns: True neu doi thanh cong.
        """
        try:
            if not new_ip:
                # Fallback: tu tim tu ipv6_rotator
                from modules.ipv6_rotator import get_ipv6_rotator
                rotator = get_ipv6_rotator()
                if not rotator or not rotator.enabled or not rotator.ipv6_list:
                    self.log("[IPv6] Khong co IPv6 de rotate", "WARN")
                    return False
                new_ip = rotator.rotate()
                if not new_ip:
                    self.log("[IPv6] Rotate failed!", "WARN")
                    return False

            old_ip = self.ipv6
            self.ipv6 = new_ip
            self.log(f"[IPv6] Rotated: {old_ip[:20]}... â†’ {new_ip[:20]}...", "OK")

            # Restart SOCKS5 proxy voi IP moi
            if self._proxy:
                try:
                    self._proxy.stop()
                except Exception:
                    pass

            # Tao proxy moi (ke ca khi truoc do khong co proxy)
            if not self._proxy_port:
                self._proxy_port = self.port + 200
            try:
                from modules.ipv6_proxy import IPv6SocksProxy
                self._proxy = IPv6SocksProxy(
                    listen_port=self._proxy_port,
                    ipv6_address=new_ip,
                    log_func=lambda msg: self.log(f"[PROXY] {msg}"),
                )
                if self._proxy.start():
                    self.log(f"[IPv6] SOCKS5 proxy restarted OK", "OK")
                else:
                    self.log(f"[IPv6] SOCKS5 proxy restart FAILED!", "ERROR")
                    return False
            except Exception as e:
                self.log(f"[IPv6] Proxy restart error: {e}", "ERROR")
                return False

            return True
        except Exception as e:
            self.log(f"[IPv6] Rotate error: {e}", "WARN")
            return False

    def restart_with_new_fingerprint(self, clear_data: bool = False) -> bool:
        """
        Restart Chrome voi fingerprint moi.
        Goi khi bi 403 lien tiep.

        Args:
            clear_data: True = xoa Chrome data (force re-login).
                        False = chi restart Chrome (giu login, nhanh hon).

        Returns: True neu restart thanh cong.
        """
        self._fingerprint_seed = _get_unique_seed()
        action = "RESTART + CLEAR DATA" if clear_data else "RESTART (giu data)"
        self.log(f"[SPOOF] === {action} voi fingerprint moi (seed={self._fingerprint_seed}) ===")

        # 1. Dong Chrome hien tai
        try:
            if self.page:
                self.page.quit()
        except Exception:
            pass
        self.page = None
        self.ready = False

        # 2. Clear Chrome data CHI KHI duoc yeu cau
        if clear_data:
            self._clear_chrome_data()
            # v1.0.532: Reset project_url khi clear data (can tao project moi sau login)
            self.project_url = None

        # 3. Setup lai (setup() se goi inject_fingerprint_spoof)
        ok = self.setup(skip_403_reset=True)
        if ok:
            self.log(f"[SPOOF] Restart OK - fingerprint moi active", "OK")
        else:
            self.log(f"[SPOOF] Restart FAIL!", "ERROR")
        return ok

    # ============================================================
    # Setup - Khá»Ÿi táº¡o Chrome + vÃ o Flow + táº¡o project
    # ============================================================

    def _clear_chrome_data(self):
        """Xoa Chrome data (giu First Run) de force fresh login."""
        import shutil
        data_dir = self.chrome_data
        if not data_dir.exists():
            return

        self.log(f"Clear Chrome data: {data_dir}")
        first_run = data_dir / "First Run"
        has_first_run = first_run.exists()

        for item in data_dir.iterdir():
            if item.name == "First Run":
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(str(item), ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception:
                pass

        # Tao lai First Run neu can
        if not has_first_run:
            try:
                first_run.touch()
            except Exception:
                pass
        self.log("Chrome data cleared", "OK")

    def _check_current_account(self) -> str:
        """
        Check email dang login trong Chrome hien tai.
        Mo myaccount.google.com -> doc email.
        Returns: email string hoac "" neu chua login.
        """
        try:
            if not self.page:
                return ""
            self.page.get("https://myaccount.google.com")
            time.sleep(3)

            current_url = self.page.url or ''
            # Neu redirect ve accounts.google.com -> chua login
            if 'accounts.google.com' in current_url:
                return ""

            # Doc email tu page
            email = self.page.run_js("""
                // Tim email trong page
                var els = document.querySelectorAll('[data-email]');
                if (els.length > 0) return els[0].getAttribute('data-email');

                // Fallback: tim text co @ trong header area
                var all = document.querySelectorAll('header *');
                for (var i = 0; i < all.length; i++) {
                    var t = all[i].textContent.trim();
                    if (t.indexOf('@') > 0 && t.indexOf('.') > 0 && t.length < 60) {
                        return t;
                    }
                }

                // Fallback 2: tim trong aria-label
                var btns = document.querySelectorAll('[aria-label*="@"]');
                if (btns.length > 0) {
                    var label = btns[0].getAttribute('aria-label');
                    var match = label.match(/[\w.-]+@[\w.-]+/);
                    if (match) return match[0];
                }

                return '';
            """)
            return str(email or '').strip().lower()
        except Exception as e:
            self.log(f"Check account error: {e}", "WARN")
            return ""

    def setup(self, skip_403_reset: bool = False) -> bool:
        """
        Full setup:
        1. Mo Chrome -> check account hien tai
        2. Neu sai account -> clear data + login dung account
        3. Vao labs.google/fx/tools/flow
        4. Tao project moi
        5. Doi textarea san sang

        Args:
            skip_403_reset: True = khong reset 403 counter (khi restart tu 403 handler)

        Returns: True neu san sang
        """
        print("\n[SETUP] Khoi tao Chrome session...")

        # 1. Check Chrome exists
        if not self.chrome_path.exists():
            self.log(f"Chrome not found: {self.chrome_path}", "ERROR")
            return False

        # 2. Mo Chrome truoc de check account
        self.log(f"Mo Chrome: {self.chrome_path}")
        self.log("Data: Portable default (khong ep profile)")
        self.log(f"Port: {self.port}")

        from DrissionPage import ChromiumPage, ChromiumOptions

        co = ChromiumOptions()
        co.set_browser_path(str(self.chrome_path))
        co.set_address(f'127.0.0.1:{self.port}')
        co.set_argument('--no-first-run')
        co.set_argument('--no-default-browser-check')
        self._apply_window_layout_options(co)

        # ProxyProvider (uu tien) hoac IPv6 truc tiep (backward compat)
        if self._proxy_provider and self._proxy_provider.is_ready():
            chrome_arg = self._proxy_provider.get_chrome_arg()
            if chrome_arg:
                self.log(f"Proxy ({self._proxy_provider.get_type()}): {self._proxy_provider.get_current_ip()}")
                co.set_argument(f'--proxy-server={chrome_arg}')
                co.set_argument('--proxy-bypass-list=<-loopback>')
                self.log(f"Proxy READY -> Chrome dung {self._proxy_provider.get_type()}", "OK")
        elif self.ipv6:
            self._proxy_port = self.port + 200
            self.log(f"IPv6: {self.ipv6}")
            self.log(f"Starting SOCKS5 proxy on 127.0.0.1:{self._proxy_port}...")
            try:
                from modules.ipv6_proxy import IPv6SocksProxy
                self._proxy = IPv6SocksProxy(
                    listen_port=self._proxy_port,
                    ipv6_address=self.ipv6,
                    log_func=lambda msg: self.log(f"[PROXY] {msg}"),
                )
                if self._proxy.start():
                    self.log(f"SOCKS5 proxy READY -> Chrome se dung IPv6", "OK")
                    co.set_argument(f'--proxy-server=socks5://127.0.0.1:{self._proxy_port}')
                    co.set_argument('--proxy-bypass-list=<-loopback>')
                else:
                    self.log(f"SOCKS5 proxy FAILED! Chrome se dung IPv4", "ERROR")
            except Exception as e:
                self.log(f"IPv6 proxy error: {e}", "ERROR")

        try:
            self.page = ChromiumPage(co)
            self.log(f"Chrome opened: {self.page.title}", "OK")
            self._enforce_window_layout()
        except Exception as e:
            self.log(f"Chrome failed: {e}", "ERROR")
            return False

        # Chan mo tab moi truoc khi thao tac UI.
        self.inject_tab_guard()

        # Inject fingerprint ngay sau khi mo Chrome
        if self._fingerprint_seed > 0:
            self.inject_fingerprint_spoof()

        # 3. Check account hien tai
        need_login = False
        if self._account:
            target_email = self._account['id'].strip().lower()
            current_email = self._check_current_account()
            self.log(f"Account hien tai: {current_email or '(chua login)'}")
            self.log(f"Account can dung: {target_email}")

            if current_email == target_email:
                self.log("Account DUNG -> khong can login lai", "OK")
            else:
                self.log("Account SAI hoac chua login -> clear data + login", "WARN")
                need_login = True

        # 4. Login neu can
        if need_login:
            try:
                self.page.quit()
            except Exception:
                pass
            self.page = None

            self._clear_chrome_data()
            login_ok = self._auto_login()
            if not login_ok:
                self.log(f"Dang nhap that bai: {self._account['id']}", "ERROR")
                return False

            co2 = ChromiumOptions()
            co2.set_browser_path(str(self.chrome_path))
            co2.set_address(f'127.0.0.1:{self.port}')
            co2.set_argument('--no-first-run')
            co2.set_argument('--no-default-browser-check')
            self._apply_window_layout_options(co2)
            if self._proxy_provider and self._proxy_provider.is_ready():
                chrome_arg = self._proxy_provider.get_chrome_arg()
                if chrome_arg:
                    co2.set_argument(f'--proxy-server={chrome_arg}')
                    co2.set_argument('--proxy-bypass-list=<-loopback>')
            elif self._proxy and self._proxy._running:
                co2.set_argument(f'--proxy-server=socks5://127.0.0.1:{self._proxy_port}')
                co2.set_argument('--proxy-bypass-list=<-loopback>')
            try:
                self.page = ChromiumPage(co2)
                self.log(f"Chrome mo lai: {self.page.title}", "OK")
                self._enforce_window_layout()
            except Exception as e:
                self.log(f"Chrome restart failed: {e}", "ERROR")
                return False

            self.inject_tab_guard()
            if self._fingerprint_seed > 0:
                self.inject_fingerprint_spoof()

        # 5. Vao Flow page
        reuse_url = None
        if self.project_url and not need_login:
            reuse_url = self.project_url
            self.log(f"[REUSE] Co project URL cu: {reuse_url}")
            self.log(f"[REUSE] Navigate thang vao project cu (khong tao moi)")

        target_url = reuse_url or FLOW_URL
        self.log(f"Vao: {target_url}")
        self.page.get(target_url)
        time.sleep(5)
        self.apply_page_zoom()
        self.inject_fingerprint_spoof()

        current_url = self.page.url or ''
        self.log(f"URL: {current_url}")

        # Check login fallback (cho truong hop khong co _account)
        if 'accounts.google.com' in current_url:
            self.log("Chua dang nhap! Clear data truoc khi login...", "WARN")
            reuse_url = None
            self.project_url = None

            try:
                if self.page:
                    self.page.quit()
            except Exception:
                pass
            self.page = None

            self._clear_chrome_data()

            login_ok = self._auto_login()
            if not login_ok:
                self.log("Dang nhap that bai!", "ERROR")
                return False

            co3 = ChromiumOptions()
            co3.set_browser_path(str(self.chrome_path))
            co3.set_address(f'127.0.0.1:{self.port}')
            co3.set_argument('--no-first-run')
            co3.set_argument('--no-default-browser-check')
            self._apply_window_layout_options(co3)
            if self._proxy_provider and self._proxy_provider.is_ready():
                chrome_arg = self._proxy_provider.get_chrome_arg()
                if chrome_arg:
                    co3.set_argument(f'--proxy-server={chrome_arg}')
                    co3.set_argument('--proxy-bypass-list=<-loopback>')
            elif self._proxy and self._proxy._running:
                co3.set_argument(f'--proxy-server=socks5://127.0.0.1:{self._proxy_port}')
                co3.set_argument('--proxy-bypass-list=<-loopback>')
            try:
                self.page = ChromiumPage(co3)
                self.log(f"Chrome mo lai sau login: {self.page.title}", "OK")
                self._enforce_window_layout()
            except Exception as e:
                self.log(f"Chrome restart failed sau login fallback: {e}", "ERROR")
                return False

            self.inject_tab_guard()
            if self._fingerprint_seed > 0:
                self.inject_fingerprint_spoof()

            self.page.get(FLOW_URL)
            time.sleep(5)
            self.apply_page_zoom()
            self.inject_fingerprint_spoof()
            current_url = self.page.url or ''

        # 6. Reuse project cu hoac tao project moi
        if '/project/' in current_url:
            # Da vao project (reuse thanh cong hoac redirect tu dong)
            if reuse_url:
                self.log(f"[REUSE] Vao lai project cu thanh cong!", "OK")
        else:
            # Chua co project â†’ tao moi
            if reuse_url:
                self.log(f"[REUSE] Project cu khong con â†’ tao project moi", "WARN")
                self.project_url = None
            success = self._create_new_project()
            if not success:
                self.log("KhÃ´ng táº¡o Ä‘Æ°á»£c project má»›i!", "ERROR")
                return False

        # 7. Äá»£i textarea
        if self._wait_for_textarea():
            self.ready = True
            self.project_url = self.page.url
            if reuse_url:
                self.log(f"READY! Reused project: {self.project_url}", "OK")
            else:
                self.log(f"READY! New project: {self.project_url}", "OK")
            return True

        # v1.0.536: Textarea khong xuat hien â†’ retry reload project cu truoc khi tao moi
        self.log("Textarea khong xuat hien - thu reload project cu...", "WARN")

        # Retry 1: Reload project URL cu (F5) - 3 lan
        current_project = self.page.url or ''
        if '/project/' in current_project:
            for retry in range(3):
                self.log(f"[RETRY {retry+1}/3] Reload project cu...")
                try:
                    self.page.get(current_project)
                    time.sleep(3)
                    self.inject_fingerprint_spoof()

                    # Dismiss popup neu co
                    self._dismiss_popups()
                    time.sleep(1)

                    if self._wait_for_textarea(timeout=15):
                        self.ready = True
                        self.project_url = self.page.url
                        self.log(f"READY! Reused project (retry {retry+1}): {self.project_url}", "OK")
                        return True
                except Exception as e:
                    self.log(f"[RETRY {retry+1}/3] Error: {e}", "WARN")

            self.log("Project cu khong phuc hoi sau 3 lan retry â†’ tao project moi", "WARN")

        # Retry 2: Bo project cu, tao project MOI
        self.project_url = None

        try:
            self.page.get(FLOW_URL)
            time.sleep(3)
            self.inject_fingerprint_spoof()
        except Exception:
            pass

        if self._click_create_with_flow():
            self.log("Clicked 'Create with Flow'", "OK")
            time.sleep(2)

        try:
            current_url = self.page.url or ''
            if '/project/' not in current_url:
                success = self._create_new_project()
                if not success:
                    self.log("Khong tao duoc project!", "ERROR")
                    return False
        except:
            pass

        if self._wait_for_textarea():
            self.ready = True
            self.project_url = self.page.url
            self.log(f"READY! New project (fallback): {self.project_url}", "OK")
            return True

        self.log("Textarea khÃ´ng xuáº¥t hiá»‡n!", "ERROR")
        return False

    def _click_create_with_flow(self) -> bool:
        """Click nÃºt 'Create with Flow' / 'Táº¡o vá»›i Flow' náº¿u cÃ³. (giá»‘ng API mode)"""
        try:
            click_result = self.page.run_js('''
                (function() {
                    var btns = document.querySelectorAll('button');
                    for (var b of btns) {
                        var text = (b.textContent || '').trim();
                        if (text.includes('Create with Flow') || text.includes('Táº¡o vá»›i Flow')) {
                            b.click();
                            return 'CLICKED';
                        }
                    }
                    var spans = document.querySelectorAll('span');
                    for (var s of spans) {
                        var text = (s.textContent || '').trim();
                        if (text.includes('Create with Flow') || text.includes('Táº¡o vá»›i Flow')) {
                            var btn = s.closest('button');
                            if (btn) { btn.click(); return 'CLICKED_VIA_SPAN'; }
                        }
                    }
                    return 'NOT_FOUND';
                })();
            ''')
            if click_result and 'CLICKED' in str(click_result):
                return True
        except:
            pass
        return False

    def _dismiss_popups(self):
        """Dismiss popup thÃ´ng bÃ¡o (Báº¯t Ä‘áº§u / Get started / Got it / TÃ´i Ä‘á»“ng Ã½). (giá»‘ng API mode)"""
        try:
            # 1. Button popups (Báº¯t Ä‘áº§u / Get started / Got it / Dismiss)
            for sel in ['tag:button@@text():Báº¯t Ä‘áº§u', 'tag:button@@text():Get started',
                        'tag:button@@text():Báº¯t Äáº§u', 'tag:button@@text():Got it',
                        'tag:button@@text():Dismiss',
                        'tag:button@@text():ÄÃ£ hiá»ƒu', 'tag:button@@text():I understand']:
                try:
                    btn = self.page.ele(sel, timeout=0.5)
                    if btn:
                        btn.click()
                        self.log(f"Dismissed popup: {sel.split(':')[-1]}")
                        time.sleep(1)
                        return True
                except:
                    continue

            # 2. Dialog "TÃ´i Ä‘á»“ng Ã½" / "Agree" / "Accept" (giá»‘ng API mode line 5746-5760)
            self.page.run_js("""
                var dialog = document.querySelector('[role="dialog"]');
                if (dialog) {
                    var btns = dialog.querySelectorAll('button');
                    for (var i = 0; i < btns.length; i++) {
                        var text = btns[i].textContent.trim();
                        if (text.indexOf('Ä‘á»“ng Ã½') > -1 || text.indexOf('Agree') > -1 ||
                            text.indexOf('Accept') > -1 || text.indexOf('ÄÃ£ hiá»ƒu') > -1 ||
                            text.indexOf('I understand') > -1) {
                            btns[i].click();
                            break;
                        }
                    }
                }
            """)

            # 3. Fallback: Click diem trong de dismiss popup (giong API mode line 1831)
            self.page.run_js('document.elementFromPoint(window.innerWidth/2, 50).click()')
        except:
            pass
        return False

    def _is_logged_out(self) -> bool:
        """Detect logout - check URL redirect ve trang login Google. (giong API mode)"""
        try:
            url = self.page.url or ''
            logout_indicators = [
                "accounts.google.com/signin",
                "accounts.google.com/v3/signin",
                "accounts.google.com/ServiceLogin",
                "accounts.google.com/AccountChooser",
            ]
            for indicator in logout_indicators:
                if indicator in url:
                    self.log(f"[LOGOUT] Detected: {indicator}", "WARN")
                    return True
        except:
            pass
        return False

    def _create_new_project(self) -> bool:
        """
        v1.0.508: Táº¡o project má»›i - COPY Y NGUYÃŠN logic tá»« API mode warmup flow.
        1 vÃ²ng láº·p 20 láº§n: check Create with Flow + Dá»± Ã¡n má»›i + dismiss popup + reload má»—i 5 láº§n.
        """
        self.log("Táº¡o project má»›i...")
        time.sleep(2)

        # VÃ²ng láº·p giá»‘ng API mode warmup (20 láº§n retry, reload má»—i 5)
        for attempt in range(20):
            # Dam bao zoom dung truoc moi lan tim/click nut.
            self.apply_page_zoom(retries=2, force_register_script=False)

            # Check URL - cÃ³ thá»ƒ Ä‘Ã£ vÃ o project rá»“i
            try:
                current_url = self.page.url or ''
                if '/project/' in current_url:
                    self.log(f"Da vao project: {current_url}", "OK")
                    return True
            except:
                pass

            # Dismiss popup (Báº¯t Ä‘áº§u / Get started / Got it)
            self._dismiss_popups()

            # TÃ¬m button "Dá»± Ã¡n má»›i" (add_2) - Ä‘Ã¢y lÃ  má»¥c tiÃªu
            try:
                btn = self.page.ele('tag:button@@text():add_2', timeout=1)
                if btn:
                    self.log(f"Clicked 'Du an moi' (attempt {attempt+1})", "OK")
                    btn.click()
                    time.sleep(3)
                    # Äá»£i vÃ o project
                    for w in range(30):
                        try:
                            if '/project/' in (self.page.url or ''):
                                self.log(f"Project created: {self.page.url}", "OK")
                                return True
                        except:
                            pass
                        time.sleep(1)
                        if w % 10 == 9:
                            self.log(f"  ... doi vao project {w+1}s")
                    # Váº«n chÆ°a vÃ o â†’ tiáº¿p tá»¥c retry
                    self.log("Click 'Du an moi' nhung chua vao project, thu lai...", "WARN")
                    continue
            except:
                pass

            # Thá»­ JS click "Dá»± Ã¡n má»›i"
            try:
                result = self.page.run_js(JS_CLICK_NEW_PROJECT)
                if result and 'CLICKED' in str(result):
                    self.log(f"Clicked 'Du an moi' JS (attempt {attempt+1})", "OK")
                    time.sleep(3)
                    for w in range(30):
                        try:
                            if '/project/' in (self.page.url or ''):
                                self.log(f"Project created: {self.page.url}", "OK")
                                return True
                        except:
                            pass
                        time.sleep(1)
                    self.log("Click 'Du an moi' JS nhung chua vao project, thu lai...", "WARN")
                    continue
            except Exception as e:
                if "ContextLost" in str(type(e).__name__) or "refresh" in str(e).lower():
                    self.log("Page dang refresh, doi...")
                    time.sleep(2)
                    try:
                        if '/project/' in (self.page.url or ''):
                            return True
                    except:
                        pass
                    continue

            # Thá»­ click "Create with Flow" náº¿u cÃ³ (giá»‘ng API mode)
            if self._click_create_with_flow():
                self.log(f"Clicked 'Create with Flow' ({attempt+1}/20)")
                time.sleep(1)
                continue  # Check láº¡i ngay

            # Reload page má»—i 5 láº§n
            if attempt > 0 and attempt % 5 == 0:
                self.log(f"Reload Flow page ({attempt}/20)...", "WARN")
                try:
                    self.page.get(FLOW_URL)
                    time.sleep(3)
                    self.apply_page_zoom(retries=3, force_register_script=False)
                except:
                    pass

            time.sleep(0.5)

        self.log("Khong tao duoc project sau 20 lan!", "ERROR")
        return False

    def _wait_for_textarea(self, timeout: int = 30) -> bool:
        """Äá»£i textarea/contenteditable xuáº¥t hiá»‡n."""
        self.log("Äá»£i textarea...")

        for i in range(timeout):
            result = self.page.run_js("""
                var ce = document.querySelector('[contenteditable="true"]');
                var ta = document.querySelector('textarea:not([class*="recaptcha"])');
                if (ce) return 'contenteditable';
                if (ta) return 'textarea';
                return 'not_found';
            """)

            if result and result != 'not_found':
                self.log(f"Input ready: {result}", "OK")
                return True

            time.sleep(1)

        return False

    # ============================================================
    # Generate Image - Táº¡o áº£nh cho khÃ¡ch
    # ============================================================

    def generate_image(self, client_bearer_token: str, client_project_id: str,
                       client_prompt: str, model_name: str = 'GEM_PIX_2',
                       aspect_ratio: str = 'IMAGE_ASPECT_RATIO_LANDSCAPE',
                       seed: int = None, image_inputs: list = None) -> dict:
        """
        Táº¡o áº£nh - giá»‘ng y há»‡t test_local_proxy.py (Ä‘Ã£ hoáº¡t Ä‘á»™ng).

        Flow (copy tá»« test):
        1. VÃ o project URL
        2. Inject interceptor (thay token + projectId)
        3. Setup image mode + model
        4. Paste prompt â†’ Ä‘á»£i 4s recaptcha â†’ Enter
        5. Chá» response â†’ tráº£ base64

        v1.0.543: Timeout = Flow bá»‹ Ä‘Æ¡/load cháº­m â†’ retry táº¡i chá»— (khÃ´ng pháº£i 403).
        Retry tá»‘i Ä‘a 2 láº§n ná»¯a (tá»•ng 3 láº§n) trÆ°á»›c khi tráº£ fail.

        Returns: { media: [...] } hoáº·c { error: "..." }
        """
        if not self.page:
            return {"error": "Chrome not initialized"}

        max_timeout_retries = 3  # v1.0.543: Retry 3 lan khi timeout (Flow bi do)
        last_result = None

        for attempt in range(max_timeout_retries):
            if attempt > 0:
                self.log(f"=== Generate Image [RETRY {attempt}/{max_timeout_retries-1}] ===")
            else:
                self.log(f"=== Generate Image ===")
            self.log(f"Token: {client_bearer_token[:20]}...{client_bearer_token[-10:]}")
            self.log(f"ProjectId: {client_project_id}")
            self.log(f"Prompt: {client_prompt[:60]}...")

            try:
                # 1. VÃ o project (giá»‘ng test step 2)
                if self.project_url:
                    self.page.get(self.project_url)
                else:
                    self.page.get(FLOW_URL)
                time.sleep(5)
                self.inject_fingerprint_spoof()

                # Äá»£i textarea
                if not self._wait_for_textarea(timeout=20):
                    self.log("Textarea not found, tao project moi...", "WARN")
                    if not self._create_new_project():
                        return {"error": "Cannot create project"}
                    if not self._wait_for_textarea(timeout=20):
                        return {"error": "Textarea not found after project creation"}
                    self.project_url = self.page.url

                # 2. Inject interceptor (reset trÆ°á»›c khi inject + retry náº¿u fail)
                self.log("Inject interceptor...")
                if image_inputs:
                    self.log(f"Reference images: {len(image_inputs)} media ID(s)")
                # Reset state trÆ°á»›c
                self.page.run_js("window.__proxyInterceptReady = false; window._response = null; window._responseError = null; window._requestPending = false;")
                time.sleep(1)  # v1.0.547: Tang 0.5â†’1s cho page xu ly

                js = build_interceptor_js(client_bearer_token, client_project_id, image_inputs)

                # v1.0.547: Retry interceptor injection toi da 3 lan
                interceptor_ok = False
                for inject_attempt in range(3):
                    try:
                        r = self.page.run_js(js)
                        self.log(f"Interceptor: {r}")
                        if r and r in ('PROXY_INTERCEPTOR_READY', 'ALREADY'):
                            interceptor_ok = True
                            break
                        # r is None hoac unexpected â†’ retry
                        self.log(f"[WARN] Interceptor inject returned: {r}, retry {inject_attempt+1}/3")
                        time.sleep(1)
                    except Exception as ie:
                        self.log(f"[WARN] Interceptor inject error: {ie}, retry {inject_attempt+1}/3")
                        time.sleep(1)

                # Verify interceptor da inject thanh cong
                if not interceptor_ok:
                    # Thu verify bang check window variable
                    try:
                        check = self.page.run_js("return window.__proxyInterceptReady === true ? 'OK' : 'FAIL';")
                        if check == 'OK':
                            self.log("[OK] Interceptor verified via window check")
                            interceptor_ok = True
                        else:
                            self.log("[ERROR] Interceptor KHONG inject duoc! Request se dung token SAI â†’ 403")
                            return {"error": "Interceptor injection failed - cannot proceed"}
                    except:
                        self.log("[ERROR] Interceptor verification failed!")
                        return {"error": "Interceptor injection failed - cannot proceed"}

                # 3. Setup Image mode + model (giá»‘ng test step 4)
                # v1.0.487: Dung _current_model_index khi da switch model do 403
                model_index = self._current_model_index if self._current_model_index > 0 else MODEL_INDEX_MAP.get(model_name, 0)
                self.log(f"Setup Image mode (model index: {model_index})...")
                js_model = JS_SELECT_MODEL.replace('MODEL_INDEX', str(model_index))
                self.page.run_js("window._modelSelectResult = 'PENDING';")
                self.page.run_js(js_model)
                time.sleep(7)

                model_result = self.page.run_js("return window._modelSelectResult;")
                self.log(f"Model result: {model_result}")

                # 4. Paste prompt (giá»‘ng test step 5)
                self.log(f"Paste prompt...")
                ok = self._paste_prompt(client_prompt)
                if not ok:
                    return {"error": "Cannot paste prompt"}

                # 5. Äá»£i recaptcha 4s â†’ Enter (giá»‘ng test step 6)
                self.log("Doi recaptcha (4s)...")
                time.sleep(4)

                from DrissionPage.common import Keys
                self.page.actions.key_down(Keys.ENTER).key_up(Keys.ENTER)
                self.log("Enter sent!")

                # 6. Chá» response (giá»‘ng test step 7)
                result = self._wait_for_response(timeout=180)

                # 7. Cleanup browser data sau má»—i request (trÃ¡nh bá»‹ Google track/flag)
                try:
                    self.page.run_js(JS_CLEANUP)
                    self.log("Cleanup browser data OK")
                except Exception as ce:
                    self.log(f"Cleanup warning: {ce}", "WARN")

                # v1.0.543: Check timeout â†’ retry tai cho (Flow bi do, khong phai 403)
                if 'error' in result:
                    err_str = str(result['error']).lower()
                    is_timeout = 'timeout' in err_str
                    is_403 = '403' in err_str
                    is_400 = '400' in err_str

                    if is_timeout and not is_403 and not is_400:
                        # Timeout = Flow bi do/load cham â†’ retry
                        last_result = result
                        if attempt < max_timeout_retries - 1:
                            self.log(f"[TIMEOUT RETRY] Flow bi do â†’ thu lai lan {attempt + 2}/{max_timeout_retries}...", "WARN")
                            time.sleep(3)  # Doi 3s truoc khi retry
                            continue
                        else:
                            self.log(f"[TIMEOUT RETRY] Het {max_timeout_retries} lan retry, tra fail", "WARN")
                            return result
                    else:
                        # 403/400/loi khac â†’ tra ve ngay, khong retry
                        return result
                else:
                    # Thanh cong!
                    if attempt > 0:
                        self.log(f"[TIMEOUT RETRY] OK sau {attempt + 1} lan!", "OK")
                    return result

            except Exception as e:
                self.log(f"Error: {e}", "ERROR")
                import traceback
                traceback.print_exc()
                return {"error": str(e)}

        # Fallback (khong nen den day)
        return last_result or {"error": "Max retries exceeded"}

    def generate_video(self, client_bearer_token: str, client_project_id: str,
                       client_prompt: str, media_id: str,
                       video_model: str = '',  # '' = let Chrome UI decide (Lite LP)
                       aspect_ratio: str = 'VIDEO_ASPECT_RATIO_LANDSCAPE',
                       seed: int = None) -> dict:
        """
        v1.0.632: Táº¡o video tá»« áº£nh (I2V) - THá»°C Sá»° chuyá»ƒn sang Video mode.

        Flow:
        1. VÃ o project URL (giá»‘ng generate_image)
        2. Click chuyá»ƒn sang Video mode (Video tab â†’ ThÃ nh pháº§n â†’ 16:9 â†’ x1 â†’ Lower Priority)
        3. Inject video interceptor (thay token + projectId + inject mediaId)
        4. Paste prompt â†’ Enter â†’ Chrome gá»­i VIDEO request (Ä‘Ãºng URL video)
        5. Interceptor thay token/projectId + inject mediaId
        6. Chá» operations response â†’ tráº£ vá» cho VM
        7. VM tá»± poll Google trá»±c tiáº¿p Ä‘á»ƒ láº¥y video URL

        Args:
            client_bearer_token: Bearer token cá»§a VM
            client_project_id: Project ID cá»§a VM
            client_prompt: Video prompt (mÃ´ táº£ chuyá»ƒn Ä‘á»™ng)
            media_id: Media ID cá»§a áº£nh (tá»« generate_image response)
            video_model: Model video I2V
            aspect_ratio: Tá»· lá»‡ video
            seed: Seed (optional)

        Returns:
            {"operations": [...]} hoáº·c {"error": "..."}
        """
        if not self.page:
            return {"error": "Chrome not initialized"}

        max_timeout_retries = 3

        for attempt in range(max_timeout_retries):
            if attempt > 0:
                self.log(f"=== Generate Video [RETRY {attempt}/{max_timeout_retries-1}] ===")
            else:
                self.log(f"=== Generate Video (I2V) ===")
            self.log(f"MediaId: {media_id[:50]}...")
            self.log(f"Prompt: {client_prompt[:60]}...")

            try:
                # 1. VÃ o project
                if self.project_url:
                    self.page.get(self.project_url)
                else:
                    self.page.get(FLOW_URL)
                time.sleep(5)
                self.inject_fingerprint_spoof()

                if not self._wait_for_textarea(timeout=20):
                    self.log("Textarea not found, tao project moi...", "WARN")
                    if not self._create_new_project():
                        return {"error": "Cannot create project"}
                    if not self._wait_for_textarea(timeout=20):
                        return {"error": "Textarea not found after project creation"}
                    self.project_url = self.page.url

                # 2. CHUYá»‚N SANG VIDEO MODE (click UI thá»±c sá»±)
                self.log("Chuyen sang Video mode (Video tab â†’ Thanh phan â†’ 16:9 â†’ x1 â†’ Lower Priority)...")
                self.page.run_js("window._videoModeResult = 'PENDING';")
                self.page.run_js(JS_SELECT_VIDEO_MODE)
                time.sleep(7)  # Doi JS async hoan thanh (~5.6s setTimeout chain)

                video_mode_result = self.page.run_js("return window._videoModeResult;")
                self.log(f"Video mode result: {video_mode_result}")

                if video_mode_result != 'SUCCESS':
                    self.log(f"[WARN] Video mode chua SUCCESS: {video_mode_result}, thu lai...", "WARN")
                    # Dong menu va thu lai
                    self.page.run_js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));")
                    time.sleep(1)
                    self.page.run_js("window._videoModeResult = 'PENDING';")
                    self.page.run_js(JS_SELECT_VIDEO_MODE)
                    time.sleep(7)
                    video_mode_result = self.page.run_js("return window._videoModeResult;")
                    self.log(f"Video mode retry result: {video_mode_result}")

                # 3. Inject VIDEO interceptor (thay token + projectId + inject mediaId)
                self.log("Inject video interceptor...")
                self.page.run_js("""
                    window.__proxyInterceptReady = false;
                    window._response = null; window._responseError = null; window._requestPending = false;
                    window._videoResponse = null; window._videoError = null; window._videoPending = false;
                """)
                time.sleep(1)

                js_video_interceptor = build_video_interceptor_js(
                    client_bearer_token, client_project_id, media_id, video_model
                )

                interceptor_ok = False
                for inject_attempt in range(3):
                    try:
                        r = self.page.run_js(js_video_interceptor)
                        self.log(f"Video interceptor: {r}")
                        if r and r in ('PROXY_INTERCEPTOR_READY', 'ALREADY'):
                            interceptor_ok = True
                            break
                        self.log(f"[WARN] Video interceptor returned: {r}, retry {inject_attempt+1}/3")
                        time.sleep(1)
                    except Exception as ie:
                        self.log(f"[WARN] Video interceptor error: {ie}, retry {inject_attempt+1}/3")
                        time.sleep(1)

                if not interceptor_ok:
                    try:
                        check = self.page.run_js("return window.__proxyInterceptReady === true ? 'OK' : 'FAIL';")
                        if check == 'OK':
                            interceptor_ok = True
                        else:
                            return {"error": "Video interceptor injection failed"}
                    except:
                        return {"error": "Video interceptor injection failed"}

                # 4. Paste video prompt
                self.log(f"Paste video prompt...")
                ok = self._paste_prompt(client_prompt)
                if not ok:
                    return {"error": "Cannot paste prompt"}

                # 5. Doi recaptcha â†’ Enter (Chrome gui VIDEO request, interceptor thay token + inject mediaId)
                self.log("Doi recaptcha (4s)...")
                time.sleep(4)

                from DrissionPage.common import Keys
                self.page.actions.key_down(Keys.ENTER).key_up(Keys.ENTER)
                self.log("Enter sent! Chrome sends VIDEO request (interceptor replaces token + injects mediaId)...")

                # 6. Cho VIDEO response (operations) tu interceptor
                result = self._wait_for_video_response(timeout=300)

                # Check timeout/error -> retry
                if 'error' in result:
                    err_str = str(result['error']).lower() if isinstance(result['error'], str) else json.dumps(result['error']).lower()
                    is_timeout = 'timeout' in err_str
                    is_403 = '403' in err_str
                    is_400 = '400' in err_str

                    if is_timeout and not is_403 and not is_400:
                        if attempt < max_timeout_retries - 1:
                            self.log(f"[TIMEOUT RETRY] Video timeout -> retry {attempt + 2}/{max_timeout_retries}...", "WARN")
                            time.sleep(3)
                            continue
                        else:
                            try:
                                self.page.run_js(JS_CLEANUP)
                            except Exception:
                                pass
                            return result
                    else:
                        try:
                            self.page.run_js(JS_CLEANUP)
                        except Exception:
                            pass
                        return result

                # 7. v1.0.684: Server tu poll Google cho den khi video XONG
                # Google co the tra {"operations": [...]} HOAC {"media": [...]}
                # Ca 2 truong hop deu can poll de cho SUCCESSFUL
                operations = result.get('operations', [])
                media_items = result.get('media', [])
                
                # Xac dinh status tu operations hoac media
                need_poll = False
                if operations:
                    op = operations[0]
                    op_status = op.get('status', '')
                    is_done = op.get('done', False)
                    if 'SUCCESSFUL' in op_status or is_done:
                        self.log(f"Video SUCCESSFUL ngay tu dau! (operations)", "OK")
                    elif 'FAILED' in op_status:
                        self.log(f"Video FAILED: {op_status}", "ERROR")
                    else:
                        need_poll = True
                        self.log(f"Video operations status: {op_status} -> can poll")
                elif media_items:
                    # Google tra media array — can convert sang operations format
                    media_status = ''
                    try:
                        media_status = media_items[0].get('mediaMetadata', {}).get('mediaStatus', {}).get('mediaGenerationStatus', '')
                    except (IndexError, AttributeError):
                        pass
                    if 'SUCCESSFUL' in media_status or 'COMPLETE' in media_status:
                        self.log(f"Video SUCCESSFUL ngay tu dau! (media)", "OK")
                    elif 'FAILED' in media_status:
                        self.log(f"Video FAILED: {media_status}", "ERROR")
                    else:
                        # SCHEDULED, PENDING, or unknown -> can poll
                        need_poll = True
                        self.log(f"Video media status: {media_status} -> convert to operations for poll")
                        
                        # v1.0.684: Convert media format -> operations format
                        # media items co dang: {mediaId, mediaMetadata: {mediaStatus: {...}}}
                        # Google poll API can: {operations: [{operation: {name: <mediaId>}, status: "..PENDING.."}]}
                        converted_ops = []
                        for mi in media_items:
                            media_id = mi.get('mediaId', '') or mi.get('name', '') or mi.get('id', '')
                            if not media_id:
                                self.log(f"Media item khong co mediaId: {json.dumps(mi)[:200]}", "WARN")
                                continue
                            scene_id = mi.get('sceneId', '')
                            converted_ops.append({
                                "operation": {"name": media_id},
                                "sceneId": scene_id,
                                "status": "MEDIA_GENERATION_STATUS_PENDING"
                            })
                        if converted_ops:
                            operations = converted_ops
                            self.log(f"Converted {len(converted_ops)} media items to operations (ids: {[o['operation']['name'][:12] for o in converted_ops]})")
                        else:
                            self.log(f"Cannot convert media to operations - no mediaId found", "WARN")
                            self.log(f"Media items: {json.dumps(media_items)[:500]}")
                            need_poll = False
                
                if need_poll and operations:
                    self.log(f"Poll Google qua Chrome ({len(operations)} ops)...")
                    poll_result = self._poll_video_status_via_chrome(
                        operations=operations,
                        bearer_token=client_bearer_token,
                        timeout=300
                    )
                    if poll_result:
                        result = poll_result

                # 8. Cleanup
                try:
                    self.page.run_js(JS_CLEANUP)
                    self.log("Cleanup browser data OK")
                except Exception as ce:
                    self.log(f"Cleanup warning: {ce}", "WARN")

                if attempt > 0:
                    self.log(f"[TIMEOUT RETRY] Video OK sau {attempt + 1} lan!", "OK")
                return result

            except Exception as e:
                self.log(f"Video error: {e}", "ERROR")
                import traceback
                traceback.print_exc()
                return {"error": str(e)}

        return {"error": "Max retries exceeded"}

    def _wait_for_video_response(self, timeout: int = 60) -> dict:
        """Chá» video response (operations) tá»« interceptor."""
        self.log(f"Chá» video response ({timeout}s)...")
        start = time.time()

        while time.time() - start < timeout:
            elapsed = time.time() - start

            state = self.page.run_js("""
                return {
                    pending: window._videoPending,
                    hasResponse: !!window._videoResponse,
                    error: window._videoError
                };
            """)

            if state and state.get('error'):
                self.log(f"VIDEO ERROR: {state['error']}", "ERROR")
                data = self.page.run_js("return window._videoResponse;")
                if data and isinstance(data, dict) and 'error' in data:
                    return {"error": data['error']}
                return {"error": state['error']}

            if state and state.get('hasResponse') and not state.get('error'):
                self.log(f"Video response sau {elapsed:.1f}s!", "OK")
                data = self.page.run_js("return window._videoResponse;")

                if isinstance(data, dict) and 'operations' in data:
                    ops = data['operations']
                    self.log(f"Got {len(ops)} video operations", "OK")
                    return data

                return data or {"error": "Empty video response"}

            elapsed_int = int(elapsed)
            if elapsed_int > 0 and elapsed_int % 10 == 0 and elapsed_int != getattr(self, '_last_video_wait_log', 0):
                self._last_video_wait_log = elapsed_int
                self.log(f"... chá» video ({elapsed_int}s/{timeout}s)")

            time.sleep(1)

        return {"error": f"Video timeout {timeout}s"}

    def _poll_video_status_via_chrome(self, operations: list, bearer_token: str,
                                      timeout: int = 300) -> dict:
        """
        v1.0.640: Poll Google video status QUA CHROME (fetch API trong browser context).
        Chrome co session hop le nen khong bi 403 captcha.
        """
        self.log(f"Poll video status qua Chrome ({timeout}s max)...")
        start = time.time()
        poll_count = 0
        poll_interval = 5

        while time.time() - start < timeout:
            elapsed = time.time() - start
            poll_count += 1

            try:
                ops_json_str = json.dumps(operations)
                token_str = bearer_token

                # v1.0.642: Fix ReferenceError: await is not defined
                # page.run_js() wraps code in sync function() => await fails
                # Solution: Use Promise + window variable, then poll for result
                poll_js = (
                    "window._pollResult = null; window._pollDone = false;"
                    "(async function() {"
                    "  try {"
                    "    var response = await fetch("
                    "      'https://aisandbox-pa.googleapis.com/v1/video:batchCheckAsyncVideoGenerationStatus',"
                    "      {"
                    "        method: 'POST',"
                    "        headers: {"
                    "          'Authorization': 'Bearer " + token_str + "',"
                    "          'Content-Type': 'application/json'"
                    "        },"
                    "        body: JSON.stringify({ operations: " + ops_json_str + " })"
                    "      }"
                    "    );"
                    "    if (!response.ok) {"
                    "      var text = '';"
                    "      try { text = await response.text(); } catch(e) {}"
                    "      window._pollResult = { _httpError: response.status, _httpText: text.substring(0, 200) };"
                    "    } else {"
                    "      window._pollResult = await response.json();"
                    "    }"
                    "  } catch(e) {"
                    "    window._pollResult = { _fetchError: e.message };"
                    "  }"
                    "  window._pollDone = true;"
                    "})();"
                    "return 'POLL_STARTED';"
                )

                self.page.run_js(poll_js)

                # Wait for async fetch to complete (max 30s)
                fetch_start = time.time()
                result = None
                while time.time() - fetch_start < 30:
                    time.sleep(1)
                    done = self.page.run_js("return window._pollDone;")
                    if done:
                        result = self.page.run_js("return window._pollResult;")
                        break

                if not result:
                    if poll_count % 6 == 1:
                        self.log(f"Video poll {poll_count}: null response ({int(elapsed)}s)")
                    time.sleep(poll_interval)
                    continue

                if '_httpError' in result:
                    http_code = result['_httpError']
                    self.log(f"Video poll {poll_count}: HTTP {http_code} ({int(elapsed)}s)")
                    if http_code == 401:
                        self.log("Bearer token het han!", "ERROR")
                        return None
                    time.sleep(poll_interval)
                    continue

                if '_fetchError' in result:
                    self.log(f"Video poll {poll_count}: fetch error: {result['_fetchError']} ({int(elapsed)}s)")
                    time.sleep(poll_interval)
                    continue

                ops = result.get('operations', [])
                if not ops:
                    if poll_count % 6 == 1:
                        self.log(f"Video poll {poll_count}: no operations ({int(elapsed)}s)")
                    time.sleep(poll_interval)
                    continue

                op = ops[0]
                op_status = op.get('status', '')
                is_done = op.get('done', False)

                if poll_count % 3 == 1 or is_done or 'SUCCESSFUL' in op_status or 'FAILED' in op_status:
                    self.log(f"Video poll {poll_count}: status={op_status}, done={is_done} ({int(elapsed)}s)")

                if 'SUCCESSFUL' in op_status or is_done:
                    self.log(f"Video SUCCESSFUL sau {int(elapsed)}s!", "OK")
                    return result

                if 'FAILED' in op_status or 'ERROR' in op_status:
                    error_msg = op.get('error', op_status)
                    self.log(f"Video FAILED: {error_msg}", "ERROR")
                    return {"operations": ops, "error": str(error_msg)}

                time.sleep(poll_interval)

            except Exception as e:
                self.log(f"Video poll {poll_count}: error {e}", "ERROR")
                time.sleep(poll_interval)

        self.log(f"Video poll timeout sau {int(time.time() - start)}s!", "ERROR")
        return None



    def _paste_prompt(self, prompt: str) -> bool:
        """Paste prompt báº±ng Ctrl+V (giá»‘ng tool)."""
        import pyperclip
        from DrissionPage.common import Keys
        with _PROMPT_PASTE_LOCK:
            self.log("[PASTE-LOCK] Acquired")

            # Focus input
            input_type = self.page.run_js("""
                var ce = document.querySelector('[contenteditable="true"]');
                var ta = document.querySelector('textarea:not([class*="recaptcha"])');
                if (ce) { ce.focus(); ce.click(); return 'contenteditable'; }
                if (ta) { ta.focus(); ta.click(); return 'textarea'; }
                return 'not_found';
            """)

            if input_type == 'not_found':
                for _ in range(5):
                    time.sleep(1)
                    input_type = self.page.run_js("""
                        var ce = document.querySelector('[contenteditable="true"]');
                        var ta = document.querySelector('textarea:not([class*="recaptcha"])');
                        if (ce) { ce.focus(); ce.click(); return 'contenteditable'; }
                        if (ta) { ta.focus(); ta.click(); return 'textarea'; }
                        return 'not_found';
                    """)
                    if input_type != 'not_found':
                        break

            if input_type == 'not_found':
                self.log("Input not found!", "ERROR")
                return False

            self.log(f"Input type: {input_type}")
            time.sleep(0.3)

            # Ctrl+A â†’ Ctrl+V
            self.page.actions.key_down(Keys.CONTROL).key_down('a').key_up('a').key_up(Keys.CONTROL)
            time.sleep(0.2)

            pyperclip.copy(prompt)
            self.page.actions.key_down(Keys.CONTROL).key_down('v').key_up('v').key_up(Keys.CONTROL)
            time.sleep(0.5)

            # Verify
            actual = self.page.run_js("""
                var ce = document.querySelector('[contenteditable="true"]');
                if (ce) return ce.textContent.substring(0, 80);
                var ta = document.querySelector('textarea:not([class*="recaptcha"])');
                if (ta) return ta.value.substring(0, 80);
                return '';
            """)
            self.log(f"Verified: '{actual}'")
            self.log("[PASTE-LOCK] Released")
            return bool(actual and actual.strip())

    def _wait_for_response(self, timeout: int = 180) -> dict:
        """Chá» response tá»« interceptor."""
        self.log(f"Chá» response ({timeout}s)...")
        start = time.time()

        while time.time() - start < timeout:
            elapsed = time.time() - start

            state = self.page.run_js("""
                return {
                    pending: window._requestPending,
                    hasResponse: !!window._response,
                    error: window._responseError
                };
            """)

            if state and state.get('error'):
                self.log(f"ERROR: {state['error']}", "ERROR")
                data = self.page.run_js("return window._response;")
                if data and isinstance(data, dict) and 'error' in data:
                    return {"error": data['error']}
                return {"error": state['error']}

            if state and state.get('hasResponse') and not state.get('error'):
                self.log(f"Response sau {elapsed:.1f}s!", "OK")
                data = self.page.run_js("return window._response;")

                if isinstance(data, dict) and 'media' in data:
                    # Download images (convert fifeUrl â†’ base64 náº¿u cáº§n)
                    media = data['media']
                    for i, item in enumerate(media):
                        gen_img = item.get('image', {}).get('generatedImage', {})
                        encoded = gen_img.get('encodedImage', '')
                        fife_url = gen_img.get('fifeUrl', '')

                        # Náº¿u khÃ´ng cÃ³ base64 nhÆ°ng cÃ³ URL â†’ download
                        if not encoded and fife_url:
                            try:
                                import requests as req
                                r = req.get(fife_url, timeout=30)
                                if r.status_code == 200:
                                    gen_img['encodedImage'] = base64.b64encode(r.content).decode()
                                    self.log(f"Image {i+1} downloaded from URL ({len(r.content):,} bytes)", "OK")
                            except Exception as e:
                                self.log(f"Download failed: {e}", "WARN")

                    self.log(f"Got {len(media)} images", "OK")
                    return data

                return data or {"error": "Empty response"}

            # Log moi 30s (tranh spam)
            elapsed_int = int(elapsed)
            if elapsed_int > 0 and elapsed_int % 30 == 0 and elapsed_int != getattr(self, '_last_wait_log', 0):
                self._last_wait_log = elapsed_int
                self.log(f"... chá» ({elapsed_int}s/{timeout}s)")

            time.sleep(1)

        return {"error": f"Timeout {timeout}s"}

    # ============================================================
    # Auto Login - Doc tai khoan tu sheet "SERVER" cot B
    # ============================================================

    def _auto_login(self) -> bool:
        """
        Tu dong dang nhap Google.

        Uu tien:
        1. Dung self._account neu da set (tu ChromePool)
        2. Fallback: doc tu sheet SERVER + xoay vong
        """
        # 1. Uu tien account da duoc assign
        if self._account:
            account = self._account
            self.log(f"Dung account da assign: {account['id']}")
        else:
            # Fallback: doc tu sheet
            self.log("Doc tai khoan tu sheet 'SERVER' cot B...")
            accounts = get_server_accounts()
            if not accounts:
                self.log("Khong tim thay tai khoan!", "ERROR")
                return False

            server_index_file = TOOL_DIR / "config" / ".server_account_index.json"
            current_index = 0
            try:
                if server_index_file.exists():
                    data = json.loads(server_index_file.read_text(encoding='utf-8'))
                    current_index = data.get('index', 0)
                    if current_index >= len(accounts):
                        current_index = 0
            except Exception:
                pass

            account = accounts[current_index]
            self._account = account
            self.log(f"Dung tai khoan {current_index + 1}/{len(accounts)}: {account['id']}")

        # 2. Goi login
        try:
            sys.path.insert(0, str(TOOL_DIR))
            from google_login import login_google_chrome, get_proxy_arg_from_settings

            # Dong Chrome hien tai truoc khi login
            if self.page:
                try:
                    self.page.quit()
                except Exception:
                    pass
                self.page = None

            # v1.0.653: Dung CUNG port voi worker (Chrome da quit, port trong)
            # Bug cu: login_port = self.port + 100 â†’ port 19322 (sai!)
            # â†’ Login mo Chrome rieng, worker khong connect duoc
            login_worker_id = self.port - 9222  # 19222 â†’ 10000, etc.

            # v1.0.653: Dung CUNG proxy voi worker (khong phai settings.yaml)
            # Bug cu: get_proxy_arg_from_settings() â†’ doc ipv6.txt â†’ SAI IPv6, pha mang
            _proxy_arg = ""
            if self._proxy_provider and self._proxy_provider.is_ready():
                _chrome_arg = self._proxy_provider.get_chrome_arg()
                if _chrome_arg:
                    _proxy_arg = _chrome_arg
                    self.log(f"Login proxy: {_proxy_arg}")
            if not _proxy_arg:
                try:
                    _proxy_arg = get_proxy_arg_from_settings()
                except Exception:
                    pass

            success = login_google_chrome(
                account_info=account,
                chrome_portable=str(self.chrome_path),
                profile_dir=None,
                worker_id=login_worker_id,  # v1.0.653: CUNG port voi worker
                proxy_arg=_proxy_arg,
            )

            if success:
                self.log(f"Dang nhap thanh cong: {account['id']}", "OK")
                # KHONG mo lai Chrome o day - setup() se mo sau
                return True
            else:
                self.log(f"Dang nhap that bai: {account['id']}", "ERROR")
                return False

        except Exception as e:
            self.log(f"Loi dang nhap: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    # ============================================================
    # Cleanup / Restart
    # ============================================================

    def restart_chrome(self):
        """Restart Chrome (clear data + reopen)."""
        self.log("Restarting Chrome...")
        self.ready = False

        try:
            if self.page:
                self.page.run_js(JS_CLEANUP)
                time.sleep(1)
        except Exception:
            pass

        # Re-setup
        self.setup()

    def close(self):
        """Close Chrome."""
        self.ready = False
        try:
            if self.page:
                self.page.quit()
        except Exception:
            pass
        self.page = None

