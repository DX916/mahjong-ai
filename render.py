import json
import os
import base64
from pathlib import Path
import webbrowser


class MahjongRenderer:
    """麻将对局可视化渲染器 - 完整版 v7"""

    TILE_DIR = "riichi-mahjong-tiles-master/Export/Regular"

    TILE_MAP = {
        'bamboo-1': 'Sou1.png', 'bamboo-2': 'Sou2.png', 'bamboo-3': 'Sou3.png',
        'bamboo-4': 'Sou4.png', 'bamboo-5': 'Sou5.png', 'bamboo-6': 'Sou6.png',
        'bamboo-7': 'Sou7.png', 'bamboo-8': 'Sou8.png', 'bamboo-9': 'Sou9.png',
        'characters-1': 'Man1.png', 'characters-2': 'Man2.png', 'characters-3': 'Man3.png',
        'characters-4': 'Man4.png', 'characters-5': 'Man5.png', 'characters-6': 'Man6.png',
        'characters-7': 'Man7.png', 'characters-8': 'Man8.png', 'characters-9': 'Man9.png',
        'dots-1': 'Pin1.png', 'dots-2': 'Pin2.png', 'dots-3': 'Pin3.png',
        'dots-4': 'Pin4.png', 'dots-5': 'Pin5.png', 'dots-6': 'Pin6.png',
        'dots-7': 'Pin7.png', 'dots-8': 'Pin8.png', 'dots-9': 'Pin9.png',
        'dragons-green': 'Hatsu.png', 'dragons-red': 'Chun.png', 'dragons-white': 'Haku.png',
        'winds-east': 'Ton.png', 'winds-south': 'Nan.png', 'winds-west': 'Shaa.png', 'winds-north': 'Pei.png',
        'man5-dora': 'Man5-Dora.png', 'pin5-dora': 'Pin5-Dora.png', 'sou5-dora': 'Sou5-Dora.png'
    }

    # 麻将牌排序优先级
    TILE_ORDER = {
        'characters': 0,  # 万子
        'bamboo': 1,  # 条子
        'dots': 2,  # 筒子
        'winds': 3,  # 风牌
        'dragons': 4,  # 三元牌
    }

    def __init__(self, replay_path='replays/latest_game.json'):
        self.replay_path = replay_path
        self.data = None

    def load_replay(self):
        """加载replay数据"""
        if not os.path.exists(self.replay_path):
            raise FileNotFoundError(f"Replay文件不存在: {self.replay_path}")

        with open(self.replay_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"✅ 加载replay成功: {len(self.data['history'])} 步")
        return self.data

    def encode_tile_images(self):
        """将麻将牌图片编码为base64"""
        tile_base64 = {}

        all_tiles = set(self.TILE_MAP.values())
        all_tiles.add('Back.png')

        for tile_file in all_tiles:
            tile_path = os.path.join(self.TILE_DIR, tile_file)
            if os.path.exists(tile_path):
                with open(tile_path, 'rb') as f:
                    tile_base64[tile_file] = base64.b64encode(f.read()).decode('utf-8')
            else:
                print(f"⚠️  图片不存在: {tile_path}")

        return tile_base64

    def generate_html(self, output_path='replay_viewer.html'):
        """生成独立的HTML可视化文件"""

        if self.data is None:
            self.load_replay()

        try:
            tile_base64 = self.encode_tile_images()
            use_embedded_images = True
        except Exception as e:
            print(f"⚠️  无法加载图片: {e}")
            print("   将使用文本占位符")
            tile_base64 = {}
            use_embedded_images = False

        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>麻将AI对局回放 - CGI-ISMCTS</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header h1 {{
            font-size: 32px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .header-controls {{
            display: flex;
            gap: 20px;
            align-items: center;
        }}

        .view-selector {{
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .view-selector select {{
            background: white;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
        }}

        .stats {{
            display: flex;
            gap: 30px;
            font-size: 14px;
        }}

        .stat-item {{
            text-align: center;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .main-content {{
            display: flex;
            gap: 20px;
            padding: 20px;
        }}

        .left-panel {{
            flex: 0 0 250px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        .right-panel {{
            flex: 0 0 280px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        .panel {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .panel-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }}

        .mahjong-table {{
            flex: 1;
            position: relative;
            background: linear-gradient(135deg, #2d5016 0%, #1a3d0f 100%);
            border-radius: 12px;
            padding: 30px;
            min-height: 900px;
            box-shadow: inset 0 0 50px rgba(0,0,0,0.3);
        }}

        .table-inner {{
            width: 100%;
            height: 100%;
            position: relative;
        }}

        .player-area {{
            position: absolute;
            display: flex;
            align-items: center;
        }}

        .player-bottom {{
            bottom: 15px;
            left: 50%;
            transform: translateX(-50%);
            flex-direction: column;
            gap: 6px;
        }}

        .player-top {{
            top: 15px;
            left: 50%;
            transform: translateX(-50%);
            flex-direction: column;
            gap: 6px;
        }}

        .player-left {{
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            flex-direction: row;
            gap: 6px;
        }}

        .player-right {{
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            flex-direction: row;
            gap: 6px;
        }}

        .player-info {{
            background: rgba(255,255,255,0.95);
            padding: 6px 12px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            text-align: center;
            z-index: 10;
        }}

        .player-name {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 2px;
            font-size: 11px;
        }}

        .player-score {{
            font-size: 14px;
            font-weight: bold;
            color: #e74c3c;
        }}

        .tiles-container {{
            display: flex;
            gap: 10px;
        }}

        .player-bottom .tiles-container,
        .player-top .tiles-container {{
            flex-direction: row;
            align-items: center;
        }}

        .player-left .tiles-container,
        .player-right .tiles-container {{
            flex-direction: column;
            align-items: center;
        }}

        .pile-section {{
            display: flex;
            gap: 3px;
            padding: 4px;
            background: rgba(52, 152, 219, 0.1);
            border-radius: 3px;
        }}

        .player-bottom .pile-section,
        .player-top .pile-section {{
            flex-direction: row;
        }}

        .player-left .pile-section,
        .player-right .pile-section {{
            flex-direction: column;
        }}

        .hand-section {{
            display: flex;
            gap: 2px;
        }}

        .player-bottom .hand-section,
        .player-top .hand-section {{
            flex-direction: row;
        }}

        .player-left .hand-section,
        .player-right .hand-section {{
            flex-direction: column;
        }}

        .tile {{
            background: white;
            border: 1px solid #ccc;
            border-radius: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            overflow: hidden;
        }}

        .player-bottom .tile,
        .player-top .tile {{
            width: 38px;
            height: 52px;
        }}

        .player-left .tile,
        .player-right .tile {{
            width: 52px;
            height: 38px;
        }}

        .tile img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}

        .tile.hidden {{
            background: linear-gradient(135deg, #c0392b 0%, #8e44ad 100%);
            color: white;
            font-size: 10px;
        }}

        .discard-center {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 600px;
            height: 600px;
            display: grid;
            grid-template-areas:
                ". top ."
                "left center right"
                ". bottom .";
            grid-template-columns: 140px 1fr 140px;
            grid-template-rows: 140px 1fr 140px;
            gap: 10px;
        }}

        .discard-section {{
            display: flex;
            flex-wrap: wrap;
            gap: 2px;
            padding: 6px;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
        }}

        .discard-bottom {{
            grid-area: bottom;
            justify-content: center;
            align-content: flex-start;
        }}

        .discard-top {{
            grid-area: top;
            justify-content: center;
            align-content: flex-end;
        }}

        /* 关键修复：左右弃牌区域支持换行 */
        .discard-left {{
            grid-area: left;
            justify-content: flex-end;
            align-content: center;
        }}

        .discard-right {{
            grid-area: right;
            justify-content: flex-start;
            align-content: center;
        }}

        .discard-tile {{
            background: white;
            border: 1px solid #ccc;
            border-radius: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        .discard-bottom .discard-tile,
        .discard-top .discard-tile {{
            width: 28px;
            height: 38px;
        }}

        .discard-left .discard-tile,
        .discard-right .discard-tile {{
            width: 38px;
            height: 28px;
        }}

        .discard-tile img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}

        .winrate-chart {{
            width: 100%;
            height: 250px;
            background: #2c3e50;
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }}

        .controls {{
            background: white;
            padding: 20px;
            border-top: 2px solid #ecf0f1;
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .timeline {{
            flex: 1;
            height: 60px;
            background: #ecf0f1;
            border-radius: 8px;
            position: relative;
            cursor: pointer;
            overflow: hidden;
        }}

        .timeline-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
            transition: width 0.3s;
            border-radius: 8px;
        }}

        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }}

        .btn-primary {{
            background: #3498db;
            color: white;
        }}

        .btn-primary:hover {{
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        }}

        .btn-secondary {{
            background: #95a5a6;
            color: white;
        }}

        .btn-secondary:hover {{
            background: #7f8c8d;
        }}

        .step-info {{
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
            min-width: 150px;
            text-align: center;
        }}

        .action-log {{
            max-height: 400px;
            overflow-y: auto;
        }}

        .action-item {{
            padding: 10px;
            margin-bottom: 8px;
            background: white;
            border-radius: 6px;
            border-left: 4px solid #3498db;
            transition: all 0.2s;
        }}

        .action-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .action-item.current {{
            background: #e8f4fd;
            border-left-color: #e74c3c;
        }}

        .player-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .player-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}

        .player-card.active {{
            border-left-color: #2ecc71;
            background: #e8f8f5;
        }}

        .player-card.viewing {{
            border-left-color: #f39c12;
            background: #fef5e7;
        }}

        ::-webkit-scrollbar {{
            width: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <span>🀄</span>
                <span>麻将AI对局回放系统</span>
            </h1>
            <div class="header-controls">
                <div class="view-selector">
                    <span>视角选择:</span>
                    <select id="viewPerspective" onchange="changeViewPerspective()">
                        <option value="0">玩家 0</option>
                        <option value="1">玩家 1</option>
                        <option value="2">玩家 2</option>
                        <option value="3">玩家 3</option>
                    </select>
                </div>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value" id="totalSteps">-</div>
                        <div>总步数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="currentStep">-</div>
                        <div>当前步</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="left-panel">
                <div class="panel">
                    <div class="panel-title">📊 玩家信息</div>
                    <div class="player-list" id="playerList"></div>
                </div>

                <div class="panel">
                    <div class="panel-title">📝 动作记录</div>
                    <div class="action-log" id="actionLog"></div>
                </div>
            </div>

            <div class="mahjong-table">
                <div class="table-inner">
                    <div class="player-area player-bottom" id="playerBottom"></div>
                    <div class="player-area player-top" id="playerTop"></div>
                    <div class="player-area player-left" id="playerLeft"></div>
                    <div class="player-area player-right" id="playerRight"></div>

                    <div class="discard-center">
                        <div class="discard-section discard-bottom" id="discardBottom"></div>
                        <div class="discard-section discard-top" id="discardTop"></div>
                        <div class="discard-section discard-left" id="discardLeft"></div>
                        <div class="discard-section discard-right" id="discardRight"></div>
                    </div>
                </div>
            </div>

            <div class="right-panel">
                <div class="panel">
                    <div class="panel-title">📈 胜率曲线</div>
                    <canvas id="winrateChart" class="winrate-chart"></canvas>
                </div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-secondary" onclick="previousStep()">◀ 上一步</button>
            <button class="btn btn-primary" id="playBtn" onclick="togglePlay()">▶ 播放</button>
            <button class="btn btn-secondary" onclick="nextStep()">下一步 ▶</button>
            <div class="step-info" id="stepInfo">Step 0</div>
            <div class="timeline" id="timeline" onclick="seekTimeline(event)">
                <div class="timeline-fill" id="timelineFill"></div>
            </div>
        </div>
    </div>

    <script>
        const replayData = {json.dumps(self.data, ensure_ascii=False)};
        const tileImages = {json.dumps(tile_base64, ensure_ascii=False)};
        const useEmbeddedImages = {json.dumps(use_embedded_images)};
        const TILE_MAP = {json.dumps(self.TILE_MAP, ensure_ascii=False)};

        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;
        let viewPerspective = 0;
        let winrateChart = null;

        // 麻将牌排序函数
        function sortMahjongTiles(tiles) {{
            const tileOrder = {{
                'characters': 0,
                'bamboo': 1,
                'dots': 2,
                'winds': 3,
                'dragons': 4
            }};

            return tiles.slice().sort((a, b) => {{
                const [typeA, numA] = a.split('-');
                const [typeB, numB] = b.split('-');

                const orderA = tileOrder[typeA] !== undefined ? tileOrder[typeA] : 99;
                const orderB = tileOrder[typeB] !== undefined ? tileOrder[typeB] : 99;

                if (orderA !== orderB) {{
                    return orderA - orderB;
                }}

                const valA = parseInt(numA) || numA;
                const valB = parseInt(numB) || numB;

                if (typeof valA === 'number' && typeof valB === 'number') {{
                    return valA - valB;
                }}

                return String(valA).localeCompare(String(valB));
            }});
        }}

        function getTileImageSrc(tileName, isHidden = false) {{
            if (!useEmbeddedImages) {{
                return '';
            }}

            if (isHidden) {{
                const backImg = tileImages['Back.png'];
                return backImg ? `data:image/png;base64,${{backImg}}` : '';
            }}

            // 关键修复：清理牌名，移除可能的特殊字符
            const cleaned = String(tileName).toLowerCase().trim().replace(/\\s+/g, '');

            // 尝试直接映射
            let fileName = TILE_MAP[cleaned];

            // 如果没找到，打印调试信息并返回空字符串
            if (!fileName) {{
                console.warn('未找到牌名映射:', tileName, '清理后:', cleaned);
                // 尝试显示原始牌名作为fallback
                return '';
            }}

            const imgData = tileImages[fileName];
            if (!imgData) {{
                console.warn('未找到图片数据:', fileName, '对应牌名:', tileName);
                return '';
            }}

            return `data:image/png;base64,${{imgData}}`;
        }}

        function renderTile(tileName, rotation = 0, isHidden = false) {{
            // 额外检查：如果牌名是空字符串或undefined，跳过
            if (!tileName || tileName === '') {{
                console.warn('空牌名');
                return '';
            }}

            const imgSrc = getTileImageSrc(tileName, isHidden);
            const hiddenClass = isHidden ? ' hidden' : '';
            const rotateStyle = rotation !== 0 ? `transform: rotate(${{rotation}}deg);` : '';

            if (imgSrc) {{
                return `<div class="tile${{hiddenClass}}"><img src="${{imgSrc}}" style="${{rotateStyle}}" alt="${{tileName}}"></div>`;
            }} else {{
                // 如果没有图片，显示调试信息
                const displayText = isHidden ? '?' : String(tileName).substring(0, 5);
                console.warn('无法渲染牌:', tileName);
                return `<div class="tile${{hiddenClass}}" style="${{rotateStyle}}; background: #ffcccc; color: #cc0000; font-size: 8px;">${{displayText}}</div>`;
            }}
        }}

        function renderDiscardTile(tileName, rotation = 0) {{
            if (!tileName || tileName === '') {{
                return '';
            }}

            const imgSrc = getTileImageSrc(tileName, false);
            const rotateStyle = rotation !== 0 ? `transform: rotate(${{rotation}}deg);` : '';

            if (imgSrc) {{
                return `<div class="discard-tile"><img src="${{imgSrc}}" style="${{rotateStyle}}" alt="${{tileName}}"></div>`;
            }} else {{
                const displayText = String(tileName).substring(0, 5);
                console.warn('无法渲染弃牌:', tileName);
                return `<div class="discard-tile" style="${{rotateStyle}}; background: #ffcccc; color: #cc0000; font-size: 8px;">${{displayText}}</div>`;
            }}
        }}

        function changeViewPerspective() {{
            viewPerspective = parseInt(document.getElementById('viewPerspective').value);
            renderStep(currentStep);
            updateWinrateChart();
        }}

        function getRelativePosition(playerIdx) {{
            return (playerIdx - viewPerspective + 4) % 4;
        }}

        function getPositionName(relPos) {{
            const names = ['Bottom', 'Right', 'Top', 'Left'];
            return names[relPos];
        }}

        function getRotationForPosition(position) {{
            const rotations = {{
                'Bottom': 0,
                'Right': -90,
                'Top': 180,
                'Left': 90
            }};
            return rotations[position] || 0;
        }}

        function initWinrateChart() {{
            const canvas = document.getElementById('winrateChart');
            const ctx = canvas.getContext('2d');

            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;

            winrateChart = {{ canvas, ctx }};
        }}

        function updateWinrateChart() {{
            if (!winrateChart) return;

            const {{ canvas, ctx }} = winrateChart;
            const width = canvas.width;
            const height = canvas.height;

            ctx.fillStyle = '#2c3e50';
            ctx.fillRect(0, 0, width, height);

            const winrateData = [];
            for (let i = 0; i <= currentStep; i++) {{
                const stepData = replayData.history[i];
                if (stepData.state_values && stepData.state_values[viewPerspective] !== null) {{
                    winrateData.push(stepData.state_values[viewPerspective]);
                }}
            }}

            if (winrateData.length === 0) {{
                ctx.fillStyle = '#ecf0f1';
                ctx.font = '14px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('暂无胜率数据', width / 2, height / 2);
                return;
            }}

            const minVal = Math.min(...winrateData);
            const maxVal = Math.max(...winrateData);
            const range = maxVal - minVal || 1;
            const normalizedData = winrateData.map(v => (v - minVal) / range);

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1;

            for (let i = 0; i <= 4; i++) {{
                const y = (height - 40) * (i / 4) + 20;
                ctx.beginPath();
                ctx.moveTo(40, y);
                ctx.lineTo(width - 20, y);
                ctx.stroke();
            }}

            ctx.fillStyle = '#ecf0f1';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'right';

            for (let i = 0; i <= 4; i++) {{
                const y = (height - 40) * (i / 4) + 20;
                const value = (1 - i / 4).toFixed(2);
                ctx.fillText(value, 35, y + 4);
            }}

            ctx.strokeStyle = '#3498db';
            ctx.lineWidth = 2;
            ctx.beginPath();

            const stepWidth = (width - 60) / Math.max(normalizedData.length - 1, 1);

            for (let i = 0; i < normalizedData.length; i++) {{
                const x = 40 + i * stepWidth;
                const y = 20 + (1 - normalizedData[i]) * (height - 40);

                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }}

            ctx.stroke();

            if (currentStep < normalizedData.length) {{
                const x = 40 + currentStep * stepWidth;
                const y = 20 + (1 - normalizedData[currentStep]) * (height - 40);

                ctx.fillStyle = '#e74c3c';
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();
            }}

            ctx.fillStyle = '#ecf0f1';
            ctx.font = 'bold 14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`玩家 ${{viewPerspective}} 胜率变化`, width / 2, 15);
        }}

        function renderStep(step) {{
            const data = replayData.history[step];
            const players = replayData.config.players;
            const payoffs = replayData.payoffs;

            const playerListHTML = players.map((p, idx) => {{
                const isActive = idx === data.player_id;
                const isViewing = idx === viewPerspective;
                let cardClass = 'player-card';
                if (isActive) cardClass += ' active';
                if (isViewing) cardClass += ' viewing';

                return `
                    <div class="${{cardClass}}">
                        <div class="player-name">${{p.name}}</div>
                        <div class="player-score">${{payoffs[idx].toFixed(2)}}</div>
                        <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                            手牌: ${{data.all_hands[idx].length}} | 副露: ${{data.all_piles[idx].length}}
                        </div>
                    </div>
                `;
            }}).join('');
            document.getElementById('playerList').innerHTML = playerListHTML;

            // 渲染四个位置的玩家
            for (let relPos = 0; relPos < 4; relPos++) {{
                const playerIdx = (viewPerspective + relPos) % 4;
                const positionName = getPositionName(relPos);
                const rotation = getRotationForPosition(positionName);

                const player = players[playerIdx];
                let hands = data.all_hands[playerIdx] || [];
                const piles = data.all_piles[playerIdx] || [];
                const isBottomPlayer = relPos === 0;

                if (isBottomPlayer) {{
                    hands = sortMahjongTiles(hands);
                }}

                // 构建手牌
                let handHTML = '<div class="hand-section">';
                hands.forEach(tile => {{
                    if (tile) {{  // 确保tile不是空值
                        handHTML += renderTile(tile, rotation, !isBottomPlayer);
                    }}
                }});
                handHTML += '</div>';

                // 构建副露
                let pileHTML = '';
                if (piles && piles.length > 0) {{
                    pileHTML = '<div class="pile-section">';
                    piles.forEach(tile => {{
                        if (tile) {{  // 确保tile不是空值
                            // 关键：副露永远不隐藏
                            pileHTML += renderTile(tile, rotation, false);
                        }}
                    }});
                    pileHTML += '</div>';
                }}

                // 构建玩家信息
                const infoHTML = `
                    <div class="player-info">
                        <div class="player-name">${{player.name}}</div>
                        <div class="player-score">${{payoffs[playerIdx].toFixed(2)}}</div>
                    </div>
                `;

                // 根据位置组织布局
                let areaHTML = '';
                if (positionName === 'Bottom') {{
                    areaHTML = `
                        ${{infoHTML}}
                        <div class="tiles-container">
                            ${{handHTML}}
                            ${{pileHTML}}
                        </div>
                    `;
                }} else if (positionName === 'Top') {{
                    areaHTML = `
                        <div class="tiles-container">
                            ${{handHTML}}
                            ${{pileHTML}}
                        </div>
                        ${{infoHTML}}
                    `;
                }} else if (positionName === 'Left') {{
                    areaHTML = `
                        ${{infoHTML}}
                        <div class="tiles-container">
                            ${{handHTML}}
                            ${{pileHTML}}
                        </div>
                    `;
                }} else {{
                    areaHTML = `
                        <div class="tiles-container">
                            ${{handHTML}}
                            ${{pileHTML}}
                        </div>
                        ${{infoHTML}}
                    `;
                }}

                document.getElementById('player' + positionName).innerHTML = areaHTML;
            }}

            // 渲染弃牌区
            for (let relPos = 0; relPos < 4; relPos++) {{
                const playerIdx = (viewPerspective + relPos) % 4;
                const positionName = getPositionName(relPos);
                const rotation = getRotationForPosition(positionName);
                const discards = data.discards[playerIdx] || [];

                let discardHTML = '';
                discards.slice(0, 24).forEach(tile => {{
                    if (tile) {{
                        discardHTML += renderDiscardTile(tile, rotation);
                    }}
                }});

                if (discardHTML === '') {{
                    discardHTML = `<div style="color: rgba(255,255,255,0.5); font-size: 11px; padding: 5px;">无</div>`;
                }}

                document.getElementById('discard' + positionName).innerHTML = discardHTML;
            }}

            const actionHTML = replayData.history.slice(0, step + 1).reverse().slice(0, 20).map((h, idx) => `
                <div class="action-item${{idx === 0 ? ' current' : ''}}">
                    <div><strong>Step ${{h.step}}</strong> - Player ${{h.player_id}}</div>
                    <div style="font-size: 12px; color: #7f8c8d; margin-top: 3px;">${{h.action_name || '动作'}}</div>
                </div>
            `).join('');
            document.getElementById('actionLog').innerHTML = actionHTML;

            document.getElementById('currentStep').textContent = step;
            document.getElementById('stepInfo').textContent = `Step ${{step}} / ${{replayData.history.length - 1}}`;
            document.getElementById('timelineFill').style.width = `${{(step / (replayData.history.length - 1)) * 100}}%`;

            updateWinrateChart();
        }}

        function nextStep() {{
            if (currentStep < replayData.history.length - 1) {{
                currentStep++;
                renderStep(currentStep);
            }} else {{
                stopPlay();
            }}
        }}

        function previousStep() {{
            if (currentStep > 0) {{
                currentStep--;
                renderStep(currentStep);
            }}
        }}

        function togglePlay() {{
            if (isPlaying) {{
                stopPlay();
            }} else {{
                startPlay();
            }}
        }}

        function startPlay() {{
            isPlaying = true;
            document.getElementById('playBtn').textContent = '⏸ 暂停';
            playInterval = setInterval(() => {{
                nextStep();
            }}, 1000);
        }}

        function stopPlay() {{
            isPlaying = false;
            document.getElementById('playBtn').textContent = '▶ 播放';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}

        function seekTimeline(event) {{
            const rect = event.target.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const percent = x / rect.width;
            currentStep = Math.floor(percent * (replayData.history.length - 1));
            renderStep(currentStep);
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') previousStep();
            if (e.key === 'ArrowRight') nextStep();
            if (e.key === ' ') {{
                e.preventDefault();
                togglePlay();
            }}
        }});

        document.getElementById('totalSteps').textContent = replayData.history.length - 1;
        initWinrateChart();
        renderStep(0);
    </script>
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML可视化文件已生成: {output_path}")
        return output_path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='麻将AI对局可视化渲染器')
    parser.add_argument('--replay', default='replays/latest_game.json', help='replay文件路径')
    parser.add_argument('--output', default='replay_viewer.html', help='输出HTML文件路径')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')

    args = parser.parse_args()

    renderer = MahjongRenderer(replay_path=args.replay)
    output_path = renderer.generate_html(output_path=args.output)

    if not args.no_browser:
        abs_path = os.path.abspath(output_path)
        webbrowser.open(f'file://{abs_path}')
        print(f"🌐 已在浏览器中打开: {abs_path}")

    print("\n使用说明:")
    print("  - 视角选择: 下拉菜单选择任意玩家视角,胜率曲线会相应切换")
    print("  - ← → 方向键: 前后步")
    print("  - 空格键: 播放/暂停")
    print("  - 点击时间轴: 跳转到指定步骤")

if __name__ == "__main__":
    main()