"""Build/read a browser smoke harness without changing saved annotations.

The harness only writes synthetic review choices to an isolated browser profile.
Run a headless browser with --dump-dom on its HTML, then use --check DOM_FILE.
"""

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r"""
<script>
(async()=>{
const results={checks:[],status:'running'};
function require(value,name){if(!value)throw Error(name);results.checks.push(name);}
const wait=async predicate=>{for(let i=0;i<100;i++){if(predicate())return;await new Promise(r=>setTimeout(r,50));}throw Error('Timed out waiting for loaded image');};
try{
await wait(()=>byId('image').complete&&byId('image').naturalWidth>0);
require(byId('position').textContent.startsWith('1 / 200'),'first image and position loaded');
const firstId=boot.selectedIds[0];
byId('blur').value='clear';byId('blur').dispatchEvent(new Event('change'));
byId('distortion').value='none';byId('distortion').dispatchEvent(new Event('change'));
byId('notes').value='AUTOMATED_UI_TEST_ONLY';byId('notes').dispatchEvent(new Event('input'));save();
byId('next').click();await wait(()=>byId('position').textContent.startsWith('2 / 200')&&byId('image').complete&&byId('image').naturalWidth>0);
byId('previous').click();await wait(()=>byId('position').textContent.startsWith('1 / 200')&&byId('image').complete&&byId('image').naturalWidth>0);
require(byId('blur').value==='clear'&&byId('distortion').value==='none'&&byId('notes').value==='AUTOMATED_UI_TEST_ONLY','review fields survive navigation');
require(JSON.parse(localStorage.getItem(boot.storageKey))[firstId].blur_manual==='clear','progress saved to isolated localStorage');
current=99;await render();require(byId('image').src.startsWith('data:image/png;base64,'),'last image of first chunk loaded');
byId('next').click();await wait(()=>byId('position').textContent.startsWith('101 / 200')&&byId('image').complete&&byId('image').naturalWidth>0);
require(byId('sampleId').textContent.includes(boot.selectedIds[100]),'second chunk loaded with matching ID');
let capturedBlob=null;
URL.createObjectURL=blob=>{capturedBlob=blob;return 'blob:automated-test';};URL.revokeObjectURL=()=>{};
HTMLAnchorElement.prototype.click=function(){};
byId('download').click();const csv=await capturedBlob.text();
require(csv.trim().split('\r\n').length===3001,'CSV exports header and all 3000 IDs');
require(csv.includes('"'+firstId+'","clear","none","AUTOMATED_UI_TEST_ONLY"'),'CSV contains edited sample');
byId('annotator').value='automated_ui_test';byId('downloadMetadata').click();const metadata=JSON.parse(await capturedBlob.text());
require(metadata.changed_ids.length===1&&metadata.changed_ids[0]===firstId&&metadata.selected_sample_count===200,'metadata records changed ID and selected sample');
require(metadata.annotator==='automated_ui_test','metadata records entered reviewer');
results.status='passed';
}catch(error){results.status='failed';results.error=error.message;}
const result=document.createElement('pre');result.id='automatedUiResult';result.textContent=JSON.stringify(results);document.body.appendChild(result);
})();
</script>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    output = ROOT / "runs/ui_checks"
    output.mkdir(parents=True, exist_ok=True)
    if args.check:
        content = args.check.read_text(encoding="utf-8-sig")
        match = re.search(r'<pre id="automatedUiResult">(.*?)</pre>', content, flags=re.S)
        if not match:
            raise RuntimeError("Browser did not finish the interactive smoke harness.")
        result = json.loads(html.unescape(match.group(1)))
        (output / "interactive_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        if result["status"] != "passed":
            raise RuntimeError(result.get("error", "UI check failed"))
    else:
        source = ROOT / "data/annotation_tool/index.html"
        content = source.read_text(encoding="utf-8")
        base = (source.parent.as_uri() + "/")
        content = content.replace('<head>', f'<head><base href="{base}">', 1)
        # Generated HTML has a combined lang/head line; explicit fallback keeps the chunks resolvable.
        if '<base ' not in content:
            content = content.replace('<head><meta', f'<head><base href="{base}"><meta', 1)
        content = content.replace('</body>', HARNESS + '</body>', 1)
        target = output / "interactive_smoke.html"
        target.write_text(content, encoding="utf-8")
        print(f"Isolated UI harness ready: {target}")


if __name__ == "__main__":
    main()
