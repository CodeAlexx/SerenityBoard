await sleep(3500);
const cb = [...document.querySelectorAll('input[type=checkbox]')].filter(e => e.offsetParent);
log('run checkboxes', cb.length);
if (cb[0]) { cb[0].click(); await sleep(2500); }
const tagBoxes = [];
for (const label of [...document.querySelectorAll('label, li, div')]) {
  const t = (label.textContent || '').trim();
  if ((/^loss\/train/.test(t) || /^lr\b/.test(t)) && label.querySelector('input[type=checkbox]') && label.offsetParent) tagBoxes.push(label.querySelector('input[type=checkbox]'));
}
const uniq = [...new Set(tagBoxes)];
log('tag checkboxes found', uniq.length);
for (const t of uniq) { if (!t.checked) t.click(); await sleep(500); }
await sleep(3000);
const charts = document.querySelectorAll('[_echarts_instance_]').length;
const canvases = document.querySelectorAll('canvas').length;
log('after selecting tags: echarts', charts, 'canvases', canvases, 'ws', (document.body.innerText.match(/Connected|Disconnected/) || [''])[0]);
await shot('native-04-scalars-selected');
document.querySelector('[data-tab="histograms"]').click(); await sleep(3000);
log('histograms tab: echarts', document.querySelectorAll('[_echarts_instance_]').length);
await shot('native-05-histograms');
document.querySelector('[data-tab="hparams"]').click(); await sleep(2000);
log('hparams text sample', document.body.innerText.replace(/\s+/g,' ').slice(0, 0));
document.querySelector('[data-tab="pr-curves"]').click(); await sleep(2500);
log('pr tab: echarts', document.querySelectorAll('[_echarts_instance_]').length);
document.querySelector('[data-tab="audio"]').click(); await sleep(2500);
log('audio tab: audio elements', document.querySelectorAll('audio').length, 'canvases', document.querySelectorAll('canvas').length);
await shot('native-06-audio');
return { charts };
