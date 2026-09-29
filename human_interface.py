import streamlit as st
import streamlit.components.v1 as components
import torch
import os
import base64
import json
import numpy as np
from PIL import Image
import rlcard
from rlcard.agents.mahjong_mcts.network import MahjongBrain
from rlcard.agents.mahjong_mcts.mcts import ISMCTS

# ==================== 配置 ====================
st.set_page_config(page_title="麻将AI对战", layout="wide")

TILE_DIR = "riichi-mahjong-tiles-master/Export/Regular"
MODEL_PATH = 'models_new/105000/checkpoint_ep105000.pth'

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
}


# ==================== 辅助函数 ====================
def normalize_card(card):
    """统一牌名"""
    if hasattr(card, 'get_str'):
        raw = card.get_str()
    else:
        raw = str(card)
    return raw.strip().lower().replace(" ", "").replace("\n", "")


@st.cache_resource
def encode_tile_images():
    """将麻将牌图片编码为base64（用于HTML渲染）"""
    tile_base64 = {}
    all_tiles = set(TILE_MAP.values())
    all_tiles.add('Back.png')

    for tile_file in all_tiles:
        tile_path = os.path.join(TILE_DIR, tile_file)
        if os.path.exists(tile_path):
            with open(tile_path, 'rb') as f:
                tile_base64[tile_file] = base64.b64encode(f.read()).decode('utf-8')

    return tile_base64


@st.cache_resource
def load_tile_image(tile_name):
    """加载单张牌的PIL Image对象（用于Streamlit显示）"""
    cleaned = normalize_card(tile_name)
    file_name = TILE_MAP.get(cleaned)
    if file_name:
        tile_path = os.path.join(TILE_DIR, file_name)
        if os.path.exists(tile_path):
            return Image.open(tile_path)
    return Image.new('RGB', (71, 96), color='lightgray')


@st.cache_resource
def load_ai_agent():
    """加载AI智能体"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pruning_net = MahjongBrain(num_actions=38).to(device)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        pruning_net.load_state_dict(state_dict)
    pruning_net.eval()

    return ISMCTS(pruning_net=pruning_net, constraint_net=None, env=None, n_simulations=0, cgi_on=False)


def sort_mahjong_tiles(tiles):
    """排序麻将牌"""
    tile_order = {'characters': 0, 'bamboo': 1, 'dots': 2, 'winds': 3, 'dragons': 4}

    def tile_key(tile):
        parts = tile.split('-')
        if len(parts) != 2:
            return (99, 0)
        tile_type, tile_num = parts
        order = tile_order.get(tile_type, 99)
        num = int(tile_num) if tile_num.isdigit() else 0
        return (order, num)

    return sorted(tiles, key=tile_key)


def get_game_state_json(env):
    """获取当前游戏状态的JSON数据"""
    players_data = []
    for i, player in enumerate(env.game.players):
        hand_tiles = [normalize_card(c) for c in player.hand]
        pile_tiles = [normalize_card(c) for meld in player.pile for c in meld]
        discards = [normalize_card(c) for p, c in env.action_recorder if p == i]

        players_data.append({
            'hand': hand_tiles,
            'pile': pile_tiles,
            'discards': discards,
            'name': f'AI {i}' if i != 0 else '人类'
        })

    return {
        'players': players_data,
        'current_player': int(st.session_state.player_id)
    }


def generate_html_view_only(game_data, tile_base64):
    """生成纯展示的麻将桌面HTML（无交互）"""
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                background: #2d5016;
                overflow: hidden;
            }}

            .mahjong-table {{
                position: relative;
                width: 100%;
                height: 750px;
                background: linear-gradient(135deg, #2d5016 0%, #1a3d0f 100%);
                box-shadow: inset 0 0 50px rgba(0,0,0,0.3);
            }}

            .player-area {{
                position: absolute;
                display: flex;
                align-items: center;
            }}

            .player-bottom {{
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                flex-direction: column;
                gap: 8px;
            }}

            .player-top {{
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                flex-direction: column;
                gap: 8px;
            }}

            .player-left {{
                left: 20px;
                top: 50%;
                transform: translateY(-50%);
                flex-direction: row;
                gap: 8px;
            }}

            .player-right {{
                right: 20px;
                top: 50%;
                transform: translateY(-50%);
                flex-direction: row;
                gap: 8px;
            }}

            .player-info {{
                background: rgba(255,255,255,0.95);
                padding: 8px 15px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                text-align: center;
                z-index: 10;
            }}

            .player-name {{
                font-weight: 600;
                color: #2c3e50;
                font-size: 14px;
            }}

            .player-name.active {{
                color: #e74c3c;
                animation: pulse 1.5s infinite;
            }}

            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.6; }}
            }}

            .tiles-container {{
                display: flex;
                gap: 10px;
            }}

            .player-bottom .tiles-container,
            .player-top .tiles-container {{
                flex-direction: row;
            }}

            .player-left .tiles-container,
            .player-right .tiles-container {{
                flex-direction: column;
            }}

            .pile-section {{
                display: flex;
                gap: 3px;
                padding: 5px;
                background: rgba(52, 152, 219, 0.15);
                border-radius: 4px;
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
                border: 2px solid #ccc;
                border-radius: 3px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                overflow: hidden;
            }}

            .player-bottom .tile,
            .player-top .tile {{
                width: 42px;
                height: 58px;
            }}

            .player-left .tile,
            .player-right .tile {{
                width: 58px;
                height: 42px;
            }}

            .tile img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}

            .tile.hidden {{
                background: linear-gradient(135deg, #c0392b 0%, #8e44ad 100%);
            }}

            .discard-center {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 500px;
                height: 500px;
                display: grid;
                grid-template-areas:
                    ". top ."
                    "left center right"
                    ". bottom .";
                grid-template-columns: 120px 1fr 120px;
                grid-template-rows: 120px 1fr 120px;
                gap: 8px;
            }}

            .discard-section {{
                display: flex;
                flex-wrap: wrap;
                gap: 2px;
                padding: 5px;
                background: rgba(0,0,0,0.25);
                border-radius: 6px;
            }}

            .discard-bottom {{ grid-area: bottom; justify-content: center; align-content: flex-start; }}
            .discard-top {{ grid-area: top; justify-content: center; align-content: flex-end; }}
            .discard-left {{ grid-area: left; justify-content: flex-end; align-content: center; }}
            .discard-right {{ grid-area: right; justify-content: flex-start; align-content: center; }}

            .discard-tile {{
                background: white;
                border: 1px solid #999;
                border-radius: 2px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 1px 2px rgba(0,0,0,0.2);
                overflow: hidden;
            }}

            .discard-bottom .discard-tile,
            .discard-top .discard-tile {{
                width: 26px;
                height: 36px;
            }}

            .discard-left .discard-tile,
            .discard-right .discard-tile {{
                width: 36px;
                height: 26px;
            }}

            .discard-tile img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}
        </style>
    </head>
    <body>
        <div class="mahjong-table">
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

        <script>
            const gameData = {json.dumps(game_data, ensure_ascii=False)};
            const tileImages = {json.dumps(tile_base64, ensure_ascii=False)};
            const TILE_MAP = {json.dumps(TILE_MAP, ensure_ascii=False)};

            function getTileImageSrc(tileName, isHidden = false) {{
                if (isHidden) {{
                    const backImg = tileImages['Back.png'];
                    return backImg ? `data:image/png;base64,${{backImg}}` : '';
                }}

                const cleaned = String(tileName).toLowerCase().trim().replace(/\\s+/g, '');
                let fileName = TILE_MAP[cleaned];

                if (!fileName || !tileImages[fileName]) return '';

                return `data:image/png;base64,${{tileImages[fileName]}}`;
            }}

            function sortMahjongTiles(tiles) {{
                const tileOrder = {{'characters': 0, 'bamboo': 1, 'dots': 2, 'winds': 3, 'dragons': 4}};

                return tiles.slice().sort((a, b) => {{
                    const [typeA, numA] = a.split('-');
                    const [typeB, numB] = b.split('-');

                    const orderA = tileOrder[typeA] !== undefined ? tileOrder[typeA] : 99;
                    const orderB = tileOrder[typeB] !== undefined ? tileOrder[typeB] : 99;

                    if (orderA !== orderB) return orderA - orderB;

                    const valA = parseInt(numA) || numA;
                    const valB = parseInt(numB) || numB;

                    if (typeof valA === 'number' && typeof valB === 'number') return valA - valB;

                    return String(valA).localeCompare(String(valB));
                }});
            }}

            function renderTile(tileName, rotation = 0, isHidden = false) {{
                if (!tileName) return '';

                const imgSrc = getTileImageSrc(tileName, isHidden);
                const hiddenClass = isHidden ? ' hidden' : '';
                const rotateStyle = rotation !== 0 ? `transform: rotate(${{rotation}}deg);` : '';

                if (imgSrc) {{
                    return `<div class="tile${{hiddenClass}}">
                        <img src="${{imgSrc}}" style="${{rotateStyle}}">
                    </div>`;
                }} else {{
                    return `<div class="tile${{hiddenClass}}" style="${{rotateStyle}}; background: #999;"></div>`;
                }}
            }}

            function renderDiscardTile(tileName, rotation = 0) {{
                if (!tileName) return '';

                const imgSrc = getTileImageSrc(tileName, false);
                const rotateStyle = rotation !== 0 ? `transform: rotate(${{rotation}}deg);` : '';

                if (imgSrc) {{
                    return `<div class="discard-tile"><img src="${{imgSrc}}" style="${{rotateStyle}}"></div>`;
                }}
                return '';
            }}

            function renderGame() {{
                const positions = ['Bottom', 'Right', 'Top', 'Left'];
                const rotations = {{'Bottom': 0, 'Right': -90, 'Top': 180, 'Left': 90}};

                positions.forEach((pos, relPos) => {{
                    const playerIdx = relPos;
                    const player = gameData.players[playerIdx];
                    const rotation = rotations[pos];
                    const isHuman = playerIdx === 0;
                    const isCurrent = gameData.current_player === playerIdx;

                    let hands = player.hand.slice();
                    if (isHuman && pos === 'Bottom') {{
                        hands = sortMahjongTiles(hands);
                    }}

                    // 玩家信息
                    let infoHTML = `<div class="player-info">
                        <div class="player-name${{isCurrent ? ' active' : ''}}">${{player.name}}</div>
                    </div>`;

                    // 手牌
                    let handHTML = '<div class="hand-section">';
                    hands.forEach(tile => {{
                        handHTML += renderTile(tile, rotation, !isHuman);
                    }});
                    handHTML += '</div>';

                    // 副露
                    let pileHTML = '';
                    if (player.pile && player.pile.length > 0) {{
                        pileHTML = '<div class="pile-section">';
                        player.pile.forEach(tile => {{
                            pileHTML += renderTile(tile, rotation, false);
                        }});
                        pileHTML += '</div>';
                    }}

                    // 组合布局
                    let areaHTML = '';
                    if (pos === 'Bottom') {{
                        areaHTML = infoHTML + `<div class="tiles-container">${{handHTML}}${{pileHTML}}</div>`;
                    }} else if (pos === 'Top') {{
                        areaHTML = `<div class="tiles-container">${{handHTML}}${{pileHTML}}</div>` + infoHTML;
                    }} else if (pos === 'Left') {{
                        areaHTML = infoHTML + `<div class="tiles-container">${{handHTML}}${{pileHTML}}</div>`;
                    }} else {{
                        areaHTML = `<div class="tiles-container">${{handHTML}}${{pileHTML}}</div>` + infoHTML;
                    }}

                    document.getElementById('player' + pos).innerHTML = areaHTML;

                    // 弃牌区
                    let discardHTML = '';
                    player.discards.slice(0, 20).forEach(tile => {{
                        discardHTML += renderDiscardTile(tile, rotation);
                    }});
                    document.getElementById('discard' + pos).innerHTML = discardHTML || '<div style="color: rgba(255,255,255,0.3); padding: 5px; font-size: 12px;">无</div>';
                }});
            }}

            renderGame();
        </script>
    </body>
    </html>
    '''
    return html


# ==================== 主程序 ====================
def main():
    st.title("🀄 麻将AI人机对战系统")

    # 初始化会话状态
    if 'env' not in st.session_state:
        st.session_state.env = rlcard.make('mahjong')
        st.session_state.agents = [None, load_ai_agent(), load_ai_agent(), load_ai_agent()]
        st.session_state.state, st.session_state.player_id = st.session_state.env.reset()
        st.session_state.game_over = False
        st.session_state.action_count = 0

    env = st.session_state.env
    state = st.session_state.state
    player_id = st.session_state.player_id

    # 控制栏
    cols = st.columns([1, 3, 1])
    with cols[0]:
        if st.button("🔄 重新开始", key="restart"):
            st.session_state.env = rlcard.make('mahjong')
            st.session_state.state, st.session_state.player_id = st.session_state.env.reset()
            st.session_state.game_over = False
            st.session_state.action_count = 0
            st.rerun()

    with cols[1]:
        current_info = "👤 你的回合 - 点击下方按钮出牌" if player_id == 0 else f"🤖 AI {player_id} 思考中..."
        st.info(current_info)

    with cols[2]:
        st.metric("回合", st.session_state.action_count)

    # 游戏结束
    if st.session_state.game_over or env.is_over():
        st.session_state.game_over = True
        st.balloons()
        st.success("🏁 游戏结束!")

        payoffs = env.get_payoffs()
        result_cols = st.columns(4)
        for i in range(4):
            with result_cols[i]:
                name = "👤 人类" if i == 0 else f"🤖 AI {i}"
                color = "🏆" if payoffs[i] > 0 else "💔"
                st.metric(f"{color} {name}", f"{payoffs[i]:.2f}")
        return

    # 渲染麻将桌面（纯展示）
    tile_base64 = encode_tile_images()
    game_data = get_game_state_json(env)
    html_content = generate_html_view_only(game_data, tile_base64)

    components.html(html_content, height=770, scrolling=False)

    st.divider()

    # 人类玩家交互区（使用Streamlit原生组件）
    if player_id == 0:
        player = env.game.players[0]
        hand_tiles = [normalize_card(c) for c in player.hand]
        hand_tiles = sort_mahjong_tiles(hand_tiles)
        legal_actions = list(state['legal_actions'].keys())

        st.subheader("🎯 你的手牌 - 点击按钮出牌")

        # 手牌按钮
        num_cols = min(len(hand_tiles), 14)
        hand_cols = st.columns(num_cols)

        for idx, tile in enumerate(hand_tiles):
            with hand_cols[idx]:
                # ✅ 修复：使用PIL Image对象而不是base64字符串
                tile_img = load_tile_image(tile)
                st.image(tile_img, width=60)

                # 出牌按钮
                action_id = idx
                if action_id in legal_actions:
                    if st.button("出牌", key=f"play_{idx}", use_container_width=True):
                        st.session_state.state, st.session_state.player_id = env.step(action_id)
                        st.session_state.action_count += 1
                        st.rerun()
                else:
                    st.button("×", key=f"invalid_{idx}", disabled=True, use_container_width=True)

        st.divider()

        # 特殊动作
        st.subheader("⚡ 特殊动作")
        action_cols = st.columns(4)
        special_actions = {34: "碰牌", 35: "吃牌", 36: "杠牌", 37: "过"}

        for idx, (action_id, action_name) in enumerate(special_actions.items()):
            with action_cols[idx]:
                if action_id in legal_actions:
                    if st.button(f"✓ {action_name}", key=f"special_{action_id}", use_container_width=True):
                        st.session_state.state, st.session_state.player_id = env.step(action_id)
                        st.session_state.action_count += 1
                        st.rerun()
                else:
                    st.button(f"✗ {action_name}", key=f"dis_{action_id}", disabled=True, use_container_width=True)

    # AI自动行动
    else:
        with st.spinner(f'🤖 AI {player_id} 正在思考...'):
            agent = st.session_state.agents[player_id]
            agent.env = env
            action, _ = agent.get_action_probs(state)

            st.session_state.state, st.session_state.player_id = env.step(action)
            st.session_state.action_count += 1
            st.rerun()


if __name__ == "__main__":
    main()