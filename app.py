import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
import time
import hashlib
from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

optuna.logging.set_verbosity(optuna.logging.WARNING)


def clear_optimization_results():
    """Xóa đồng thời mô hình, dự đoán và thông tin của lần tối ưu cũ."""
    for key in ('results', 'result_config', 'result_y_test', 'X_test_cols', 'opt_method'):
        st.session_state.pop(key, None)


st.set_page_config(page_title="Hệ thống Tối ưu hóa HPO - Fraud Detection", layout="wide")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>RANDOM FOREST FRAUD DETECTION</h1>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("📂 1. Nạp dữ liệu")
data_file = st.sidebar.file_uploader("Tải file dữ liệu gốc (creditcard.csv):", type=["csv"])

if data_file is not None:
    
    data_bytes = data_file.getvalue()
    data_hash = hashlib.sha256(data_bytes).hexdigest()
    df_raw = pd.read_csv(BytesIO(data_bytes))
    
    sample_size = st.sidebar.slider("Chọn số lượng dòng để tối ưu (tránh quá tải):", 1000, len(df_raw), min(5000, len(df_raw)), step=1000)
    
    opt_seed = st.sidebar.number_input("Random Seed (Hạt giống ngẫu nhiên):", min_value=0, max_value=9999, value=42, step=1)
    
    df = df_raw.sample(n=sample_size, random_state=opt_seed)

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 2. Cấu hình Threshold")
    threshold = st.sidebar.slider("Ngưỡng phát hiện gian lận (Threshold):", 0.0, 1.0, 0.20, 0.01)

    st.sidebar.success(f"Đã tải {len(df)} dòng dữ liệu!")

    
    tab1, tab2, tab3 = st.tabs(["⚙️ Dữ liệu & Tiền xử lý", "🚀 Tối ưu hóa (Optuna)", "📊 Trực quan hóa & So sánh"])

    
    with tab1:
        st.subheader("Tiền xử lý và Feature Engineering")
        
        
        df = df.dropna()
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Is_Night'] = df['Hour'].apply(lambda x: 1 if x < 6 else 0)
        df['High_Amount'] = df['Amount'].apply(lambda x: 1 if x > 1000 else 0)
        
        bins = [0, 100, 500, 1000, 5000, 2000000]
        labels = [0, 1, 2, 3, 4]
        df['Amount_Bins'] = pd.cut(df['Amount'], bins=bins, labels=labels, include_lowest=True).astype(int)
        df['High_Amount_at_Night'] = df['Is_Night'] * df['High_Amount']
        
        st.write("✅ Đã tạo các đặc trưng mới: `Hour`, `Is_Night`, `High_Amount`, `Amount_Bins`, `High_Amount_at_Night`")
        
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=opt_seed, stratify=y)
        
        st.session_state['X_train'] = X_train
        st.session_state['y_train'] = y_train
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

        st.write("✅ Cân bằng dữ liệu bằng trọng số lớp (`class_weight='balanced_subsample'`):")
        
        col1, col2 = st.columns(2)
        col1.metric("Số lượng mẫu huấn luyện (Train - 80%)", len(X_train))
        col2.metric("Số lượng mẫu kiểm thử (Test - 20%)", len(X_test))

    with tab2:
        st.subheader("Chạy thuật toán Tối ưu hóa Siêu tham số (Optuna)")
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            method = st.selectbox(
                "Phương pháp tối ưu:", 
                ["Chạy & So sánh cả hai (TPE vs Random Search)", "Bayesian Optimization (TPE)", "Random Search"]
            )
        with col_opt2:
            n_trials = st.number_input("Số lần lặp (Trials):", min_value=10, max_value=200, value=20, step=10)

        current_config = {
            'data_hash': data_hash,
            'sample_size': sample_size,
            'seed': opt_seed,
            'method': method,
            'n_trials': n_trials,
        }
        if ('results' in st.session_state
                and st.session_state.get('result_config') != current_config):
            clear_optimization_results()
            st.info("Cấu hình đã thay đổi. Kết quả cũ đã được xóa; hãy nhấn Bắt đầu tối ưu hóa để chạy lại.")

        if st.button(f"🚀 Bắt đầu tối ưu hóa ({n_trials} trials, Seed = {opt_seed})"):
            
            clear_optimization_results()
            X_tr = st.session_state['X_train']
            y_tr = st.session_state['y_train']
            X_te = st.session_state['X_test']
            y_te = st.session_state['y_test']

            def create_objective(seed_val):
                def objective(trial):
                    rf_n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
                    rf_max_depth = trial.suggest_int('max_depth', 2, 8)
                    rf_min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                    rf_min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
                    rf_max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                    
                    model = RandomForestClassifier(
                        n_estimators=rf_n_estimators,
                        max_depth=rf_max_depth,
                        min_samples_split=rf_min_samples_split,
                        min_samples_leaf=rf_min_samples_leaf,
                        max_features=rf_max_features,
                        class_weight='balanced_subsample',
                        random_state=seed_val,
                        n_jobs=-1
                    )
                    
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_val)
                    score = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc').mean()
                    return score
                return objective

            def run_single_opt(sampler_type):
                if sampler_type == "TPE":
                    sampler = optuna.samplers.TPESampler(seed=opt_seed, n_startup_trials=10)
                else:
                    sampler = optuna.samplers.RandomSampler(seed=opt_seed)
                
                study = optuna.create_study(direction='maximize', sampler=sampler)
                start_time = time.time()
                study.optimize(create_objective(opt_seed), n_trials=n_trials)
                elapsed_time = time.time() - start_time
                
                best_model = RandomForestClassifier(
                    **study.best_params, 
                    class_weight='balanced_subsample', 
                    random_state=opt_seed,
                    n_jobs=-1
                )
                best_model.fit(X_tr, y_tr)
                
                t0 = time.time()
                y_prob = best_model.predict_proba(X_te)[:, 1]
                t1 = time.time()
                latency = ((t1 - t0) * 1000 / len(X_te)) * 1000
                
                return study, best_model, elapsed_time, y_prob, latency

            with st.spinner("Đang chạy tối ưu hóa... Vui lòng chờ trong giây lát."):
                results_dict = {}
                
                if "So sánh" in method or "TPE" in method:
                    st.write("🔄 Đang chạy Bayesian Optimization (TPE)...")
                    study_tpe, model_tpe, time_tpe, prob_tpe, lat_tpe = run_single_opt("TPE")
                    results_dict["TPE"] = {
                        "study": study_tpe, "model": model_tpe, "time": time_tpe, 
                        "prob": prob_tpe, "latency": lat_tpe, "params": study_tpe.best_params,
                        "best_score": study_tpe.best_value
                    }
                
                if "So sánh" in method or "Random" in method:
                    st.write("🔄 Đang chạy Random Search...")
                    study_rand, model_rand, time_rand, prob_rand, lat_rand = run_single_opt("Random")
                    results_dict["Random"] = {
                        "study": study_rand, "model": model_rand, "time": time_rand, 
                        "prob": prob_rand, "latency": lat_rand, "params": study_rand.best_params,
                        "best_score": study_rand.best_value
                    }

                st.session_state['results'] = results_dict
                st.session_state['result_config'] = current_config.copy()
                st.session_state['result_y_test'] = y_te.copy()
                st.session_state['X_test_cols'] = X_te.columns.copy()
                st.session_state['opt_method'] = method
                st.success("Tối ưu hóa hoàn tất!")

        if 'results' in st.session_state:
            results_dict = st.session_state['results']
            saved_config = st.session_state['result_config']
            st.caption(
                f"Kết quả lần chạy: Seed = {saved_config['seed']} | "
                f"Trials = {saved_config['n_trials']} cho mỗi phương pháp | "
                f"Số dòng lấy mẫu = {saved_config['sample_size']}"
            )

            fig_conv, ax_conv = plt.subplots(figsize=(10, 4.5))
            
            for m_name, res in results_dict.items():
                history = [t.value for t in res["study"].trials if t.value is not None]
                convergence = np.maximum.accumulate(history)
                color = '#2E86C1' if m_name == "TPE" else '#E67E22'
                ax_conv.plot(range(1, len(convergence) + 1), convergence, marker='o', label=f"{m_name} (Best AUC: {res['best_score']:.4f})", color=color, lw=2)
            
            ax_conv.set_title(f"Biểu đồ Hội tụ Tối ưu hóa - Convergence Curve (Seed = {opt_seed})")
            ax_conv.set_xlabel("Số lần lặp (Trial)")
            ax_conv.set_ylabel("ROC-AUC Score CV (5-Fold)")
            ax_conv.grid(True)
            ax_conv.legend(loc="lower right")
            st.pyplot(fig_conv)
            plt.close(fig_conv)

            st.write("🔥 **Bộ tham số tốt nhất thu được:**")
            for m_name, res in results_dict.items():
                st.markdown(f"**{m_name}:** (ROC-AUC CV: `{res['best_score']:.4f}` - Thời gian: `{res['time']:.2f}s`)")
                st.json(res["params"])

    with tab3:
        if 'results' in st.session_state:
            results_dict = st.session_state['results']
            y_test_curr = st.session_state['result_y_test']
            
            st.subheader("📊 Bảng So sánh Hiệu năng giữa các Phương pháp")
            
            comp_rows = []
            for m_name, res in results_dict.items():
                y_prob = res["prob"]
                roc_auc_val = auc(*roc_curve(y_test_curr, y_prob)[:2])
                precision, recall, _ = precision_recall_curve(y_test_curr, y_prob)
                pr_auc_val = auc(recall, precision)
                
                comp_rows.append({
                    "Phương pháp": m_name,
                    "ROC-AUC CV (5-Fold)": f"{res['best_score']:.4f}",
                    "ROC-AUC Test": f"{roc_auc_val:.4f}",
                    "PR-AUC Test": f"{pr_auc_val:.4f}",
                    "Thời gian chạy (s)": f"{res['time']:.2f}s",
                    "Độ trễ suy luận": f"{res['latency']:.2f} ms / 1k mẫu"
                })
            
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
            st.markdown("---")
            
            selected_m = st.radio("Chọn mô hình để xem chi tiết biểu đồ:", list(results_dict.keys()), horizontal=True)
            
            res_sel = results_dict[selected_m]
            y_prob_curr = res_sel["prob"]
            best_model_curr = res_sel["model"]
            y_pred_curr = (y_prob_curr >= threshold).astype(int)

            st.subheader(f"Phân tích Chi tiết Mô hình: {selected_m}")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Ngưỡng quyết định (Threshold)", f"{threshold:.2f}")
            col_m2.metric("Độ trễ suy luận", f"{res_sel['latency']:.2f} ms / 1k mẫu")

            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                fpr, tpr, _ = roc_curve(y_test_curr, y_prob_curr)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'Đường cong ROC ({selected_m})')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
                plt.close(fig_roc)

            with col_viz2:
                precision, recall, _ = precision_recall_curve(y_test_curr, y_prob_curr)
                pr_auc = auc(recall, precision)
                fig_pr, ax_pr = plt.subplots(figsize=(6, 4.5))
                ax_pr.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
                ax_pr.set_xlabel('Recall')
                ax_pr.set_ylabel('Precision')
                ax_pr.set_title(f'Đường cong Precision-Recall ({selected_m})')
                ax_pr.legend(loc="lower left")
                st.pyplot(fig_pr)
                plt.close(fig_pr)

            col_viz3, col_viz4 = st.columns(2)
            
            with col_viz3:
                cm = confusion_matrix(y_test_curr, y_pred_curr)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_title(f'Ma trận nhầm lẫn (Threshold = {threshold:.2f})')
                ax_cm.set_ylabel('Thực tế')
                ax_cm.set_xlabel('Dự đoán')
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            with col_viz4:
                importances = best_model_curr.feature_importances_
                indices = np.argsort(importances)[::-1][:10]
                features = st.session_state['X_test_cols']
                
                fig_fi, ax_fi = plt.subplots(figsize=(6, 4.5))
                sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis', ax=ax_fi)
                ax_fi.set_title('Top 10 Đặc trưng Quan trọng Nhất')
                st.pyplot(fig_fi)
                plt.close(fig_fi)
                
            joblib.dump(best_model_curr, f"best_rf_model_{selected_m}.pkl")
            with open(f"best_rf_model_{selected_m}.pkl", "rb") as file:
                st.download_button(
                    label=f"📥 Tải mô hình {selected_m} (.pkl)", 
                    data=file, 
                    file_name=f"best_rf_model_{selected_m}.pkl", 
                    mime="application/octet-stream"
                )
        else:
            st.info("Vui lòng chuyển sang Tab 2 và nhấn nút 'Bắt đầu tối ưu hóa' để huấn luyện và so sánh mô hình!")
else:
    clear_optimization_results()
    for key in ('X_train', 'y_train', 'X_test', 'y_test'):
        st.session_state.pop(key, None)
    st.info("Vui lòng tải file dữ liệu `creditcard.csv` ở thanh bên (Sidebar) để bắt đầu.")
