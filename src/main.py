"""CodeKnowledge - 智能代码仓库理解系统。"""
import os, sys
from dotenv import load_dotenv
load_dotenv()  # Load .env into os.environ (required for HF_ENDPOINT etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from src.pipeline import CodeKnowledgePipeline
from src.utils.logger import get_logger
logger = get_logger(__name__)
st.set_page_config(page_title='CodeKnowledge - 智能代码仓库理解系统', page_icon='\U0001f50d', layout='wide', initial_sidebar_state='expanded')
for key in ['pipeline', 'messages', 'repo_ingested', 'index_stats']:
    if key not in st.session_state:
        st.session_state[key] = {} if key == 'index_stats' else (None if key == 'pipeline' else (False if key == 'repo_ingested' else []))
def init_pipeline():
    if st.session_state.pipeline is None:
        st.session_state.pipeline = CodeKnowledgePipeline()
    return st.session_state.pipeline
with st.sidebar:
    st.title('CodeKnowledge')
    st.caption('智能代码仓库理解系统')
    st.divider()
    repo_url = st.text_input('GitHub 仓库地址', placeholder='https://github.com/user/repo', help='输入公开 GitHub 仓库的 URL')
    branch = st.text_input('分支（可选）', placeholder='main', help='留空则使用默认分支')
    col1, col2 = st.columns(2)
    with col1: ingest_btn = st.button('分析仓库', type='primary', use_container_width=True)
    with col2: reset_btn = st.button('重置', use_container_width=True)
    if ingest_btn and repo_url:
        with st.status('正在分析仓库...', expanded=True) as status:
            st.write('正在克隆仓库...')
            try:
                stats = init_pipeline().ingest(repo_url, branch=branch or None)
                if stats.get('status') == 'success':
                    st.success(f"已分析：{stats['repo_name']}")
                    st.write(f"代码块：{stats['chunks_count']}")
                    st.write(f"文件数：{stats['files_count']}")
                    st.write(f"符号数：{stats['symbols_count']}")
                    st.write(f"语言：{', '.join(stats['languages'])}")
                    st.session_state.repo_ingested = True
                    st.session_state.index_stats = stats
                    st.session_state.messages = []
                    status.update(label='分析完成！', state='complete')
                else:
                    st.warning(stats.get('message', '未找到代码'))
                    status.update(label='分析警告', state='warning')
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                logger.error("Ingestion failed:\n%s", tb)
                st.error(f"分析失败：{e}")
                st.code(tb)  # Show full traceback in UI
                status.update(label='分析失败', state='error')
    if "show_graph" not in st.session_state:
        st.session_state.show_graph = False
    if st.session_state.repo_ingested:
        st.divider()
        if st.button("?????", use_container_width=True):
            st.session_state.show_graph = not st.session_state.show_graph
    if reset_btn:
        if st.session_state.pipeline: st.session_state.pipeline.reset()
        st.session_state.pipeline = None
        st.session_state.messages = []
        st.session_state.repo_ingested = False
        st.session_state.index_stats = {}
        st.rerun()
    st.divider()
    st.caption('使用说明')
    st.markdown('1. 输入 GitHub 仓库地址\n2. 点击分析仓库\n3. 等待索引构建完成\n4. 向代码库提问')
st.title('CodeKnowledge')
if not st.session_state.messages:
    if not st.session_state.repo_ingested:
        st.info('在侧边栏输入 GitHub 仓库地址，点击分析仓库开始使用。')
    else:
        st.success('仓库分析完成！可以开始提问了。')
        s = st.session_state.index_stats
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric('代码块', s.get('chunks_count', 0))
        with c2: st.metric('文件数', s.get('files_count', 0))
        with c3: st.metric('函数/类', s.get('symbols_count', 0))
        with c4: st.metric('语言数', len(s.get('languages', [])))
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
if st.session_state.repo_ingested:
    query = st.chat_input('输入你的问题...')
    if query:
        st.session_state.messages.append({'role': 'user', 'content': query})
        with st.chat_message('user'):
            st.markdown(query)
        with st.chat_message('assistant'):
            with st.spinner('正在分析...'):
                try:
                    resp = st.session_state.pipeline.ask(query)
                    st.markdown(resp)
                    st.session_state.messages.append({'role': 'assistant', 'content': resp})
                except Exception as e:
                    st.error(f"错误：{e}")
