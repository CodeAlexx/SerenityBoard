await sleep(4000);
const tabs = [...document.querySelectorAll('[data-tab]')].map(e => e.dataset.tab);
const runs = [...document.querySelectorAll('#run-list li, .run-item, [class*=run-name], [data-run]')].filter(e => e.offsetParent).map(e => e.textContent.trim().slice(0, 30));
const wsStatus = document.body.innerText.match(/connected|connecting|disconnected/i);
const canvases = document.querySelectorAll('canvas').length;
const charts = document.querySelectorAll('[_echarts_instance_]').length;
log('tabs', tabs.length, tabs.slice(0, 12));
log('runs visible', runs.slice(0, 6));
log('ws status text', wsStatus && wsStatus[0], '| canvases', canvases, '| echarts instances', charts);
await shot('native-01-scalars');
const hist = document.querySelector('[data-tab="histograms"]'); if (hist) { hist.click(); await sleep(2500); await shot('native-02-histograms'); }
const audio = document.querySelector('[data-tab="audio"]'); if (audio) { audio.click(); await sleep(2500); await shot('native-03-audio'); }
return { tabs: tabs.length, canvases, charts, wsStatus: wsStatus && wsStatus[0], runs: runs.length };
