function make_diona_figs(varargin)
%MAKE_DIONA_FIGS  Rebuild every plot from the Diona six-arm comparative page as .fig files.
%
%   make_diona_figs()
%   make_diona_figs('OutDir','figs','SavePNG',true)
%
%   Reads the campaign extract (data.json, 252 arm-cells) and writes ONE .fig per
%   panel, 54 in total, into OutDir. Figures are built invisible by default and
%   closed after saving; open one with openfig('figs/fig01_rel_l2__piston_sobol.fig').
%
%   NAME-VALUE OPTIONS
%     'DataFile'  path to data.json           (default: ../data.json, i.e. paper/data.json)
%     'OutDir'    output directory            (default: ./figs beside this file)
%     'SavePNG'   also write a 300-dpi .png   (default: false)
%     'Visible'   'on' to watch them build    (default: 'off')
%
%   THE NINE SETS (54 panels)
%     fig01_rel_l2            7   grouped bars, all six arms, log y
%     fig02_interval_width    7   grouped bars, the three interval arms, log y
%     fig03_coverage          7   grouped bars, linear y from 50%, 95% reference
%     fig04_coverage_vs_width 7   scatter, x reversed so bottom-left is the good corner
%     fig05_inference_cost    7   mean s/query bars, value labels, no marker overlay
%     fig06_diona_stage_split 7   solver stages stacked to 100% of the mean
%     fig07_stage_heatmaps    4   stage cost by function x density, shared log shading
%     fig08_stage_vs_dim      1   stage cost against input dimension, 500D tier
%     fig09_training_time     7   training time bars, value labels, log y
%
%   CONVENTIONS CARRIED OVER FROM THE PAGE, all of which are deliberate:
%     - Bars are the NOISELESS cell; the '+' overlay is the SAME cell at 20 dB.
%       On fig02 both are MEDIAN relative width; on fig05/fig09 there is no
%       overlay at all, because training time and mean query cost are one number
%       per cell rather than a distribution.
%     - Within an arm the three bars run left to right in increasing density
%       tier, shading light to dark.
%     - Panels are ordered by INPUT DIMENSION, not cell id: piston (d=7) holds
%       frozen ids 37-42 and would otherwise sort after rosenbrock (d=8). The
%       Ishigami grid block leads, being a layout variant of the d=3 function
%       rather than a dimension of its own.
%     - Axis decades are FIXED per set, not fitted per panel, so panels saved to
%       separate files stay comparable by eye. Same for the one log colour scale
%       shared across all four heat panels. fig04 is the exception and widens per
%       panel by design; its axis labels say which frame it is on.
%     - Diona is drawn as '0' on fig09: it has no training phase at all, which a
%       log axis cannot show.
%
%   Requires R2019b or newer (tiledlayout, jsondecode).

%% ------------------------------------------------------------------ options
here = fileparts(mfilename('fullpath'));
p = inputParser;
p.addParameter('DataFile', fullfile(here,'..','data.json'), @(s)ischar(s)||isstring(s));
p.addParameter('OutDir',   fullfile(here,'figs'),      @(s)ischar(s)||isstring(s));
p.addParameter('SavePNG',  false, @(x)islogical(x)||isnumeric(x));
p.addParameter('Visible',  'off', @(s)any(strcmpi(s,{'on','off'})));
p.parse(varargin{:});
opt = p.Results;
opt.OutDir = char(opt.OutDir);
if ~exist(opt.OutDir,'dir'); mkdir(opt.OutDir); end

S = struct();                      % shared style + lookup tables
S.opt   = opt;
S.arms  = {'diona','nn','gp','nsm','rbf','idw'};
S.ivl   = {'diona','gp','nsm'};    % the arms that produce an interval
S.col   = containers.Map( ...
    {'diona','nn','gp','nsm','rbf','idw'}, ...
    {hex2rgb('1f6fb2'), hex2rgb('7a4fa3'), hex2rgb('1d7a3e'), ...
     hex2rgb('c0392b'), hex2rgb('a8770c'), hex2rgb('7d4f2e')});
S.ink   = hex2rgb('12181e');
S.slate = hex2rgb('5c6873');
S.rule  = hex2rgb('dbe1e6');
S.warn  = hex2rgb('8a6a15');

% Solver stages of Algorithm 2 (manuscript), label and the timing key of each.
% The keys are the columns of the :diona timing sidecar
% (test/ml/harness/arms/diona.jl): neighbour selection (Algorithm 1, three
% passes), the L_H LP of Section 4.1, assembly of Xi psi <= b, and the two
% bound LPs (11a)-(11b). 'residual' is wall time no stage timer claims.
S.stages = { ...
    'neighbour selection',   {'neighbor'}; ...
    'L_H estimate (LP)',     {'lh'};       ...
    'assembly',              {'assembly'}; ...
    'bounds (2 LPs)',        {'bounds'};   ...
    'unattributed',          {'residual'}};
S.stageAlpha = [0.34 0.56 0.72 0.92 0.18];
% Four DISTINCT line styles + markers for the trend plot. The page separates
% these by opacity, which here rides on an undocumented per-line alpha; if a
% release rejects it the style and marker still tell the four stages apart.
S.stageDash  = {':','-','--','-.'};
S.stageMark  = {'o','s','^','d'};

R = loadRows(opt.DataFile);
fprintf('loaded %d arm-cells from %s\n', numel(R), opt.DataFile);
warnIfClamped(R);

n = 0;
n = n + setRelL2(R,S);
n = n + setWidth(R,S);
n = n + setCoverage(R,S);
n = n + setTradeoff(R,S);
n = n + setInferenceCost(R,S);
n = n + setStageSplit(R,S);
n = n + setHeatmaps(R,S);
n = n + setTrend(R,S);
n = n + setTrainingTime(R,S);

fprintf('\nwrote %d .fig files to %s\n', n, opt.OutDir);
end

%% =================================================================== loading
function R = loadRows(f)
% jsondecode gives a struct array when every object carries the same field set,
% which the extractor guarantees, and a cell array otherwise. Handle both rather
% than depending on it.
if ~exist(f,'file')
    error('make_diona_figs:noData', ...
        ['data file not found: %s\n' ...
         'Point --DataFile at the extractor output for the campaign you want.'], f);
end
raw = jsondecode(fileread(f));
if isstruct(raw) && isfield(raw,'rows'); raw = raw.rows; end
if iscell(raw)
    names = {};
    for i = 1:numel(raw); names = union(names, fieldnames(raw{i})); end
    R = repmat(cell2struct(repmat({[]},numel(names),1), names, 1), numel(raw), 1);
    for i = 1:numel(raw)
        fn = fieldnames(raw{i});
        for k = 1:numel(fn); R(i).(fn{k}) = raw{i}.(fn{k}); end
    end
else
    R = raw(:);
end
need = {'arm','name','layout','tier','dim','cell_id','noise_db','rel_l2', ...
        'coverage','median_rel_width','mean_s_query','fit_time_s','stages','n_train'};
missing = need(~isfield(R, need));
if ~isempty(missing)
    error('make_diona_figs:schema','data.json is missing field(s): %s', strjoin(missing,', '));
end
end

function warnIfClamped(R)
% The page clamps anything outside its fixed decades to the axis edge, silently.
% These figures use the same decades, so say so rather than drawing a bar that
% is quietly the wrong height.
chk = { 'rel_l2',           -4, 0, @(r)true; ...
        'median_rel_width', -3, 1, @(r)ismember(r.arm,{'diona','gp','nsm'}); ...
        'mean_s_query',     -6, 1, @(r)true; ...
        'fit_time_s',       -5, 5, @(r)true};
for i = 1:size(chk,1)
    key = chk{i,1}; lo = 10^chk{i,2}; hi = 10^chk{i,3}; keep = chk{i,4};
    v = [];
    for k = 1:numel(R)
        if keep(R(k)) && ~isempty(R(k).(key)) && R(k).(key) > 0
            v(end+1) = R(k).(key); %#ok<AGROW>
        end
    end
    if isempty(v); continue; end
    if min(v) < lo || max(v) > hi
        warning('make_diona_figs:clamped', ...
            ['%s spans %.4g..%.4g but the axis is 10^%d..10^%d -- ' ...
             'values outside are drawn at the axis edge.'], ...
            key, min(v), max(v), chk{i,2}, chk{i,3});
    end
end
end

%% ============================================================ panel ordering
function keys = panelKeys(R)
% Panels run by input dimension; the Ishigami grid block leads at rank -1.
k = arrayfun(@(r) sprintf('%s|%s', r.name, r.layout), R, 'UniformOutput', false);
keys = unique(k, 'stable');
rank = zeros(size(keys));
for i = 1:numel(keys)
    if strcmp(keys{i},'ishigami|grid'); rank(i) = -1;
    else
        f = find(strcmp(k, keys{i}), 1);
        rank(i) = R(f).dim;
    end
end
[~, ord] = sort(rank);
keys = keys(ord);
end

function s = subsetOf(R, key)
k = arrayfun(@(r) sprintf('%s|%s', r.name, r.layout), R, 'UniformOutput', false);
s = R(strcmp(k, key));
[~, o] = sort([s.cell_id]); s = s(o);
end

function t = tiersOf(s)
[~, o] = sort([s.cell_id]); s = s(o);
t = unique({s.tier}, 'stable');
end

function nm = panelTitle(key)
parts  = strsplit(key, '|');
titles = containers.Map( ...
    {'ishigami','park91a','friedman1','hartmann6','rosenbrock','piston'}, ...
    {'Ishigami','Park91a','Friedman1','Hartmann6','Rosenbrock','Piston'});
if titles.isKey(parts{1}); base = titles(parts{1});
else; base = [upper(parts{1}(1)) parts{1}(2:end)]; end
if strcmp(parts{2},'grid'); nm = [base ' - grid'];
else;                       nm = [base ' - Sobol'''];  end
end

%% ============================================================== bar-set core
function nOut = barSet(R, S, cfg)
% One panel per function: arms along x, one bar per density tier within each
% arm, plus an optional '+' overlay for the same cell at 20 dB.
keys = panelKeys(R);
nOut = 0;
for i = 1:numel(keys)
    s = subsetOf(R, keys{i});
    if isempty(s); continue; end
    tiers = tiersOf(s);
    [f, ax] = newPanel(S, panelTitle(keys{i}), cfg.ylab, ...
        sprintf('d=%d - %s', s(1).dim, cfg.sub));

    if cfg.linear
        ylim(ax, [cfg.lo cfg.hi]);
        yticks(ax, cfg.lo:cfg.step:cfg.hi);
        yticklabels(ax, arrayfun(@(v) sprintf('%.0f%%', v*100), ...
            cfg.lo:cfg.step:cfg.hi, 'UniformOutput', false));
        if isfield(cfg,'ref') && ~isempty(cfg.ref)
            yline(ax, cfg.ref, '--', sprintf('%.0f%%', cfg.ref*100), ...
                'Color', S.warn, 'LabelHorizontalAlignment','right', ...
                'LabelVerticalAlignment','bottom', 'FontSize',8, 'LineWidth',0.9);
        end
        ybase = cfg.lo;
    else
        set(ax, 'YScale', 'log');
        ylim(ax, [10^cfg.lo 10^cfg.hi]);
        yticks(ax, 10.^(cfg.lo:cfg.hi));
        ybase = 10^cfg.lo;
    end

    nA = numel(cfg.arms); nT = numel(tiers);
    gw = 1; bw = min(0.74/nT, 0.26);
    for ai = 1:nA
        a  = cfg.arms{ai};
        gx = ai;
        if isfield(cfg,'zero') && any(strcmp(cfg.zero, a))
            text(ax, gx, ybase, '0', 'HorizontalAlignment','center', ...
                'VerticalAlignment','bottom', 'FontSize',9, 'Color',S.slate);
            continue
        end
        pad = (gw - bw*nT)/2;
        for ti = 1:nT
            bv = pickVal(s, a, tiers{ti}, cfg.bar.noise, cfg.bar.key);
            x0 = gx - gw/2 + pad + (ti-1)*bw;
            cx = x0 + bw/2;
            if ~isempty(bv) && isfinite(bv) && bv > 0
                patch(ax, [x0 x0+bw x0+bw x0], [ybase ybase bv bv], S.col(a), ...
                    'FaceAlpha', 0.42 + 0.29*(ti-1), 'EdgeColor','none');
                if isfield(cfg,'barLabels') && cfg.barLabels
                    text(ax, cx, bv, [' ' vlab(bv)], 'Rotation',90, 'FontSize',7, ...
                        'Color',S.slate, 'HorizontalAlignment','left', ...
                        'VerticalAlignment','middle');
                end
            end
            if isfield(cfg,'plus') && ~isempty(cfg.plus)
                pv = pickVal(s, a, tiers{ti}, cfg.plus.noise, cfg.plus.key);
                if ~isempty(pv) && isfinite(pv) && pv > 0
                    plot(ax, cx, pv, '+', 'Color', S.ink, 'MarkerSize',6, 'LineWidth',1.3);
                end
            end
        end
    end
    xlim(ax, [0.5 nA+0.5]);
    xticks(ax, 1:nA); xticklabels(ax, cfg.arms);
    legendForBars(ax, S, cfg, tiers);
    nOut = nOut + savePanel(f, S, cfg.prefix, keys{i});
end
end

function v = pickVal(s, arm, tier, noise, key)
m = strcmp({s.arm}, arm) & strcmp({s.tier}, tier) & [s.noise_db] == noise;
if ~any(m); v = []; return; end
r = s(find(m,1));
v = r.(key);
if isempty(v); v = []; end
end

function legendForBars(ax, S, cfg, tiers)
% Arms are already named on the x axis, so the legend only has to explain the
% shading ramp and what the '+' means -- the two things the axis cannot say.
h = gobjects(0); lbl = {};
for ti = 1:numel(tiers)
    h(end+1) = patch(ax, nan(1,3), nan(1,3), S.ink, ...
        'FaceAlpha', 0.42+0.29*(ti-1), 'EdgeColor','none'); %#ok<AGROW>
    lbl{end+1} = tiers{ti}; %#ok<AGROW>
end
if isfield(cfg,'plus') && ~isempty(cfg.plus)
    h(end+1) = plot(ax, NaN, NaN, '+', 'Color', S.ink, 'MarkerSize',6, 'LineWidth',1.3);
    lbl{end+1} = cfg.plusLabel;
end
lg = legend(ax, h, lbl, 'Location','northeast', 'FontSize',7.5);
lg.Box = 'off'; lg.ItemTokenSize = [10 10];
end

%% ================================================================= the sets
function n = setRelL2(R, S)
n = barSet(R, S, struct( ...
    'prefix','fig01_rel_l2', 'arms',{S.arms}, 'lo',-4, 'hi',0, 'linear',false, ...
    'bar',struct('key','rel_l2','noise',0), ...
    'plus',struct('key','rel_l2','noise',20), ...
    'plusLabel','20 dB', 'ylab','relative L2', 'sub','bars noiseless, + at 20 dB'));
end

function n = setWidth(R, S)
n = barSet(R, S, struct( ...
    'prefix','fig02_interval_width', 'arms',{S.ivl}, 'lo',-3, 'hi',1, 'linear',false, ...
    'bar',struct('key','median_rel_width','noise',0), ...
    'plus',struct('key','median_rel_width','noise',20), ...
    'plusLabel','20 dB', 'ylab','median relative interval width', ...
    'sub','median width; bars noiseless, + at 20 dB'));
end

function n = setCoverage(R, S)
n = barSet(R, S, struct( ...
    'prefix','fig03_coverage', 'arms',{S.ivl}, 'lo',0.5, 'hi',1.0, 'linear',true, ...
    'step',0.1, 'ref',0.95, ...
    'bar',struct('key','coverage','noise',0), ...
    'plus',struct('key','coverage','noise',20), ...
    'plusLabel','20 dB', 'ylab','coverage', ...
    'sub','gp at 2\sigma; bars noiseless, + at 20 dB'));
end

function n = setInferenceCost(R, S)
n = barSet(R, S, struct( ...
    'prefix','fig05_inference_cost', 'arms',{S.arms}, 'lo',-6, 'hi',1, 'linear',false, ...
    'barLabels',true, 'bar',struct('key','mean_s_query','noise',0), 'plus',[], ...
    'ylab','mean seconds per query', 'sub','means, noiseless cells'));
end

function n = setTrainingTime(R, S)
n = barSet(R, S, struct( ...
    'prefix','fig09_training_time', 'arms',{S.arms}, 'lo',-5, 'hi',5, 'linear',false, ...
    'barLabels',true, 'zero',{{'diona'}}, ...
    'bar',struct('key','fit_time_s','noise',0), 'plus',[], ...
    'ylab','training time (s)', 'sub','diona has no training phase'));
end

%% ------------------------------------------------- coverage against width
function n = setTradeoff(R, S)
% 18 markers per panel: 3 arms x 3 tiers x 2 noise levels. X runs BACKWARDS,
% 100% on the left, so the good corner is bottom-left: high coverage, narrow
% band. Plotted conventionally the best result would sit top-right while the
% log-width axis has worse at the top, and the picture would read against itself.
keep = arrayfun(@(r) ismember(r.arm, S.ivl) && ~isempty(r.coverage) && ...
    ~isempty(r.median_rel_width) && isfinite(r.coverage) && ...
    isfinite(r.median_rel_width), R);
Rw = R(keep);
keys = panelKeys(Rw);
msz  = [6 9 13];
n = 0;
for i = 1:numel(keys)
    s = subsetOf(Rw, keys{i});
    % Tier index is resolved WITHIN the panel. A global tier list would run to
    % six entries -- the grid block uses 10^3/20^3/50^3, not 500D/1000D/2000D --
    % and index past the end of the three marker sizes.
    panelT = tiersOf(s);
    w = [s.median_rel_width]; c = [s.coverage];

    % Default frame width 0.01-1.0 against coverage 100-90%; a panel whose data
    % leaves it is widened to the next decade or 5-point step rather than
    % clipped, because dropping a marker would hide exactly the result worth
    % seeing. Panels are therefore NOT all on one frame.
    YLO = min(-2, floor(log10(min(w))));
    YHI = max( 0,  ceil(log10(max(w))));
    XLO = min(0.90, floor(min(c)/0.05)*0.05);
    XHI = 1.00;

    [f, ax] = newPanel(S, panelTitle(keys{i}), 'median relative interval width', ...
        sprintf('d=%d - %d markers - o noiseless, + 20 dB', s(1).dim, numel(s)));
    set(ax, 'YScale','log', 'XDir','reverse');
    ylim(ax, [10^YLO 10^YHI]); yticks(ax, 10.^(YLO:YHI));
    xlim(ax, [XLO XHI]);
    xt = XLO:0.05:XHI;
    xticks(ax, xt);
    xticklabels(ax, arrayfun(@(v) sprintf('%.0f%%', v*100), xt, 'UniformOutput', false));
    xline(ax, 0.95, '--', 'Color', S.warn, 'LineWidth', 0.9);
    xlabel(ax, 'coverage  (100% at the left - bottom-left is best)');

    for k = 1:numel(s)
        r  = s(k);
        ti = find(strcmp(panelT, r.tier), 1); if isempty(ti); ti = 2; end
        col = S.col(r.arm);
        if r.noise_db == 0
            plot(ax, r.coverage, r.median_rel_width, 'o', 'MarkerEdgeColor',col, ...
                'MarkerSize', msz(ti), 'LineWidth', 1.8);
        else
            plot(ax, r.coverage, r.median_rel_width, '+', 'Color', col, ...
                'MarkerSize', msz(ti)+1, 'LineWidth', 1.8);
        end
    end
    h = gobjects(0); lbl = {};
    for a = 1:numel(S.ivl)
        h(end+1) = plot(ax, NaN, NaN, 'o', 'MarkerEdgeColor', S.col(S.ivl{a}), ...
            'MarkerSize', 8, 'LineWidth', 1.8); %#ok<AGROW>
        lbl{end+1} = S.ivl{a}; %#ok<AGROW>
    end
    lg = legend(ax, h, lbl, 'Location','southwest', 'FontSize',7.5);
    lg.Box = 'off'; lg.ItemTokenSize = [10 10];
    n = n + savePanel(f, S, 'fig04_coverage_vs_width', keys{i});
end
end

%% ------------------------------------------------- diona stage split (100%)
function n = setStageSplit(R, S)
% Shares of the MEAN, not the median: the stages are additive in the mean and
% not in the median, so the segments sum to the total exactly. Normalised to
% 100% on a linear axis because the inference-cost chart is logarithmic, where a
% stacked segment's height is NOT proportional to its value -- neighbour
% selection is ~3% of the time but would draw ~45% of the bar. The absolute mean
% total is printed under each bar, so normalising loses nothing.
keep = arrayfun(@(r) strcmp(r.arm,'diona') && ~isempty(r.stages) && ...
    isstruct(r.stages) && r.noise_db == 0, R);
Rs = R(keep);
keys = panelKeys(Rs);
n = 0;
for i = 1:numel(keys)
    s = subsetOf(Rs, keys{i});
    [f, ax] = newPanel(S, panelTitle(keys{i}), 'share of mean per-query time', ...
        sprintf('d=%d - noiseless cells - mean total printed under each bar', s(1).dim));
    ylim(ax, [0 1]); yticks(ax, 0:0.2:1);
    yticklabels(ax, {'0%','20%','40%','60%','80%','100%'});

    nS = size(S.stages,1);
    hSeg = gobjects(nS,1);
    xlab = cell(1, numel(s));
    for ri = 1:numel(s)
        tot = s(ri).stages.total;
        xlab{ri} = s(ri).tier;
        if ~(tot > 0); continue; end
        cum = 0;
        for si = 1:nS
            v = stageVal(s(ri).stages, S.stages{si,2});
            if isempty(v) || ~isfinite(v) || v <= 0; continue; end
            f0 = cum/tot; f1 = (cum+v)/tot; cum = cum + v;
            hSeg(si) = patch(ax, ri + 0.3*[-1 1 1 -1], [f0 f0 f1 f1], ...
                S.col('diona'), 'FaceAlpha', S.stageAlpha(si), 'EdgeColor','none');
        end
        % The absolute mean total rides in the tick label rather than as free
        % text below the axes, which the axes box would clip.
        xlab{ri} = sprintf('%s\n%s', s(ri).tier, g4(tot));
    end
    xlim(ax, [0.4 numel(s)+0.6]);
    xticks(ax, 1:numel(s)); xticklabels(ax, xlab);
    ok = isgraphics(hSeg);
    lg = legend(ax, hSeg(ok), S.stages(ok,1), 'Location','eastoutside', 'FontSize',7.5);
    lg.Box = 'off'; lg.ItemTokenSize = [10 10];
    n = n + savePanel(f, S, 'fig06_diona_stage_split', keys{i});
end
end

%% ------------------------------------------------------------- heat panels
function n = setHeatmaps(R, S)
% ONE LOG COLOUR SCALE ACROSS ALL FOUR PANELS, so a cell reads against the other
% panels and not just its own: the stage means span 4.4 decades, hopeless on a
% linear ramp. Every cell carries its value, so the shade never has to be
% decoded precisely. The Ishigami GRID block is excluded -- its tiers are
% 10^3/20^3/50^3 rather than 500D/1000D/2000D, so it has no column to sit in.
keep = arrayfun(@(r) strcmp(r.arm,'diona') && ~isempty(r.stages) && ...
    isstruct(r.stages) && r.noise_db == 0 && strcmp(r.layout,'sobol'), R);
Rh = R(keep);
if isempty(Rh); n = 0; return; end

[~, o] = sort([Rh.dim]);
fnames = unique({Rh(o).name}, 'stable');
fdims  = arrayfun(@(k) Rh(find(strcmp({Rh.name}, fnames{k}),1)).dim, 1:numel(fnames));
tiers  = tiersOf(Rh);

LO = -2; HI = 2.5;                       % log10 milliseconds, shared
shade = @(v) 0.07 + 0.48*max(0, min(1, (log10(v)-LO)/(HI-LO)));

n = 0;
for si = 1:4
    [f, ax] = newPanel(S, S.stages{si,1}, '', 'mean ms/query - shared log shading');
    nR = numel(fnames); nC = numel(tiers);
    for ri = 1:nR                        % ishigami (d=3) at the BOTTOM
        for ci = 1:nC
            m = strcmp({Rh.name}, fnames{ri}) & strcmp({Rh.tier}, tiers{ci});
            v = [];
            if any(m)
                v = stageVal(Rh(find(m,1)).stages, S.stages{si,2});
            end
            if isempty(v) || ~isfinite(v) || v <= 0; continue; end
            v = v * 1000;                % ms
            patch(ax, ci + 0.5*[-1 1 1 -1], ri + 0.5*[-1 -1 1 1], S.col('diona'), ...
                'FaceAlpha', shade(v), 'EdgeColor','none');
            text(ax, ci, ri, vlab(v), 'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', 'FontSize',8.5, 'Color', S.ink);
        end
    end
    xlim(ax, [0.5 nC+0.5]); ylim(ax, [0.5 nR+0.5]);
    xticks(ax, 1:nC); xticklabels(ax, tiers);
    yticks(ax, 1:nR);
    yticklabels(ax, arrayfun(@(k) sprintf('%s (d=%d)', fnames{k}, fdims(k)), ...
        1:nR, 'UniformOutput', false));
    set(ax, 'YGrid','off', 'XGrid','off', 'Box','on');
    n = n + savePanel(f, S, 'fig07_stage_heatmaps', S.stages{si,1});
end
end

%% ------------------------------------------------- stage cost vs dimension
function n = setTrend(R, S)
% 500D tier only: the heat panels show that tripling the sample count at fixed
% dimension barely moves any stage, so density is not the variable worth a line.
% Every tick d=3..8 now carries a function -- piston filled d=7.
keep = arrayfun(@(r) strcmp(r.arm,'diona') && ~isempty(r.stages) && ...
    isstruct(r.stages) && r.noise_db == 0 && strcmp(r.layout,'sobol') && ...
    strcmp(r.tier,'500D'), R);
Rt = R(keep);
if isempty(Rt); n = 0; return; end
[~, o] = sort([Rt.dim]); Rt = Rt(o);

[f, ax] = newPanel(S, 'Stage cost vs input dimension', 'mean ms per query', ...
    '500D tier, noiseless Sobol'' cells');
set(ax, 'YScale','log');
d = [Rt.dim];
h = gobjects(4,1);
for si = 1:4
    y = arrayfun(@(r) stageVal(r.stages, S.stages{si,2})*1000, Rt);
    h(si) = plot(ax, d, y, 'LineStyle', S.stageDash{si}, 'Color', S.col('diona'), ...
        'LineWidth', 1.8, 'Marker', S.stageMark{si}, 'MarkerSize',5, ...
        'MarkerFaceColor', S.col('diona'), 'MarkerEdgeColor','none');
    % Per-line alpha via a 4th Color element is undocumented; if a release
    % rejects it the lines still differ by dash pattern, which carries the
    % distinction on its own.
    try %#ok<TRYNC>
        h(si).Color(4) = max(0.45, S.stageAlpha(si));
    end
    text(ax, d(end)+0.12, y(end), strrep(S.stages{si,1},' (2 LPs)',''), ...
        'FontSize',8, 'Color', S.slate, 'VerticalAlignment','middle');
end
xlim(ax, [min(d)-0.3 max(d)+1.9]);
xticks(ax, min(d):max(d));
xlabel(ax, 'input dimension d');
ylim(ax, [10^-2 10^2.5]);
lg = legend(ax, h, strrep(S.stages(1:4,1),' (2 LPs)',''), 'Location','northwest', 'FontSize',7.5);
lg.Box = 'off'; lg.ItemTokenSize = [16 10];
n = savePanel(f, S, 'fig08_stage_vs_dim', 'all');
end

%% ================================================================= plumbing
function v = stageVal(st, keys)
v = []; t = 0; any_ = false;
if isempty(st) || ~isstruct(st); return; end
for i = 1:numel(keys)
    if isfield(st, keys{i})
        x = st.(keys{i});
        if ~isempty(x) && isfinite(x); t = t + x; any_ = true; end
    end
end
if any_; v = t; end
end

function [f, ax] = newPanel(S, ttl, ylab, sub)
f = figure('Visible', S.opt.Visible, 'Color','w', 'Units','pixels', ...
           'Position', [100 100 620 400]);
t = tiledlayout(f, 1, 1, 'Padding','compact', 'TileSpacing','compact');
ax = nexttile(t);
hold(ax, 'on'); grid(ax, 'on');
ax.GridColor = S.rule; ax.GridAlpha = 1; ax.Layer = 'top';
ax.XColor = S.slate; ax.YColor = S.slate; ax.FontSize = 9;
ax.Box = 'off';
% subtitle() is R2020b+. On R2019b-R2020a fall back to a two-line title rather
% than dropping the line that says what the panel actually plots.
if isempty(sub)
    title(ax, ttl, 'FontSize', 12, 'FontWeight','bold', 'Color', S.ink);
elseif exist('subtitle', 'file') == 2 || exist('subtitle', 'builtin') == 5
    title(ax, ttl, 'FontSize', 12, 'FontWeight','bold', 'Color', S.ink);
    subtitle(ax, sub, 'FontSize', 8.5, 'Color', S.slate);
else
    title(ax, {ttl, sub}, 'FontSize', 10, 'FontWeight','normal', 'Color', S.ink);
end
if ~isempty(ylab); ylabel(ax, ylab, 'FontSize', 9.5); end
end

function n = savePanel(f, S, prefix, key)
name = sprintf('%s__%s', prefix, sanitize(key));
savefig(f, fullfile(S.opt.OutDir, [name '.fig']));
if S.opt.SavePNG
    png = fullfile(S.opt.OutDir, [name '.png']);
    if exist('exportgraphics', 'file') == 2 || exist('exportgraphics', 'builtin') == 5
        exportgraphics(f, png, 'Resolution', 300);   % R2020a+
    else
        print(f, png, '-dpng', '-r300');              % older releases
    end
end
if strcmpi(S.opt.Visible,'off'); close(f); end
fprintf('  %s.fig\n', name);
n = 1;
end

function s = sanitize(s)
% Tier labels carry superscripts (10^3 as a single glyph) and panel keys carry
% '|', neither of which belongs in a filename.
s = char(s);
s = strrep(s, char(178), '2'); s = strrep(s, char(179), '3');
s = strrep(s, '|', '_');
s = lower(regexprep(s, '[^A-Za-z0-9]+', '_'));
s = regexprep(s, '^_+|_+$', '');
end

function rgb = hex2rgb(h)
h = char(h);
rgb = [hex2dec(h(1:2)) hex2dec(h(3:4)) hex2dec(h(5:6))] / 255;
end

function s = vlab(v)
% Compact in-cell / on-bar label, matching the page's formatter.
if isempty(v) || ~isfinite(v); s = ''; return; end
if v == 0; s = '0'; return; end
a = abs(v);
if     a >= 1000; s = regexprep(sprintf('%.*f', 1*(a<10000), v/1000), '\.0$', ''); s = [s 'k'];
elseif a >= 10;   s = sprintf('%.0f', v);
elseif a >= 1;    s = sprintf('%.1f', v);
elseif a >= 0.01; s = sprintf('%.2f', v);
else;             s = sprintf('%.3g', v);
end
end

function s = g4(v)
if isempty(v) || ~isfinite(v); s = '-'; return; end
if v >= 1e4 || (v > 0 && v < 1e-3); s = sprintf('%.2e', v);
else; s = sprintf('%.4g', v); end
s = [s ' s'];
end
