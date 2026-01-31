import os
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")


class Visualizer:
    """
    数据可视化类，负责将分析结果生成为各类统计图表并保存到本地。衔接前两个analyzer的api。
    """

    def __init__(self, output_dir: str = "./output", dpi: int = 150):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    def _save_fig(self, fig, name: str) -> str:
        """保存图表到文件并返回文件路径"""
        filepath = os.path.join(self.output_dir, f"{name}.png")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        print(f"图表已保存: {filepath}")
        plt.close(fig)
        return filepath

    def plot_commit_timeline(
        self,
        commits_df: pd.DataFrame,
        title: str = "Commit Timeline",
        save_name: str = "commit_timeline",
    ) -> str:
        """绘制提交时间线图，展示按月统计的提交趋势"""
        fig, ax = plt.subplots(figsize=(14, 5))
        df = commits_df.copy()
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month").size().reset_index(name="commits")
        monthly["month"] = monthly["month"].dt.to_timestamp()
        ax.fill_between(
            monthly["month"], monthly["commits"], alpha=0.3, color="#2196F3"
        )
        ax.plot(monthly["month"], monthly["commits"], color="#1976D2", linewidth=2)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Monthly Commits", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def plot_contributor_heatmap(
        self,
        commits_df: pd.DataFrame,
        title: str = "Contributor Activity Heatmap",
        save_name: str = "contributor_heatmap",
    ) -> str:
        """绘制贡献者提交热力图，展示最活跃几年的每周每天的提交活跃度"""
        df = commits_df.copy()
        df["week"] = df["date"].dt.isocalendar().week
        df["year"] = df["date"].dt.year
        df["weekday"] = df["date"].dt.weekday
        # 选取提交最活跃的五年进行展示，避免展示过多稀疏的年份
        year_counts = df["year"].value_counts()
        target_years = year_counts.head(5).index.sort_values().tolist()
        df = df[df["year"].isin(target_years)]

        fig, axes = plt.subplots(
            len(df["year"].unique()), 1, figsize=(15, 2 * len(df["year"].unique()))
        )
        if len(df["year"].unique()) == 1:
            axes = [axes]

        for idx, year in enumerate(sorted(df["year"].unique())):
            ax = axes[idx]
            year_data = df[df["year"] == year]
            heatmap = np.zeros((7, 53))
            for _, row in (
                year_data.groupby(["week", "weekday"])
                .size()
                .reset_index(name="count")
                .iterrows()
            ):
                week = min(int(row["week"]) - 1, 52)
                heatmap[int(row["weekday"]), week] = row["count"]
            ax.imshow(
                heatmap,
                cmap=plt.cm.Greens,
                aspect="auto",
                vmin=0,
                vmax=max(1, heatmap.max()),
            )
            ax.set_ylabel(str(year), fontsize=12, fontweight="bold")
            ax.set_yticks(range(7))
            ax.set_yticklabels(
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8
            )
            ax.set_xticks([])

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def plot_author_distribution(
        self,
        author_stats: pd.DataFrame,
        top_n: int = 15,
        title: str = "Top Contributors",
        save_name: str = "author_distribution",
    ) -> str:
        """绘制贡献者分布图，按提交数和代码行数对比展示头部几个贡献者的贡献程度"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        top = author_stats.head(top_n)
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))

        ax1.barh(range(len(top)), top["commits"], color=colors)
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels(top["author"], fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel("Commits", fontsize=11)
        ax1.set_title("By Commit Count", fontsize=12)

        ax2.barh(range(len(top)), top["total_lines"], color=colors)
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels(top["author"], fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Lines Changed", fontsize=11)
        ax2.set_title("By Lines Changed", fontsize=12)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def plot_weekly_pattern(
        self,
        weekday_stats: pd.DataFrame,
        hour_stats: pd.DataFrame,
        title: str = "Development Habits",
        save_name: str = "weekly_pattern",
    ) -> str:
        """绘制开发习惯图，展示一周各天及一天各小时的提交分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors_week = ["#E8F5E9" if i < 5 else "#FFEBEE" for i in range(7)]
        ax1.bar(
            weekday_stats["weekday"],
            weekday_stats["commits"],
            color=colors_week,
            edgecolor="#4CAF50",
        )
        ax1.set_xlabel("Day of Week", fontsize=11)
        ax1.set_ylabel("Commits", fontsize=11)
        ax1.set_title("Commits by Day of Week", fontsize=12)
        plt.sca(ax1)
        plt.xticks(rotation=45, ha="right")

        ax2.bar(
            hour_stats["hour"],
            hour_stats["commits"],
            color="#90CAF9",
            edgecolor="#2196F3",
        )
        ax2.set_xlabel("Hour of Day (UTC)", fontsize=11)
        ax2.set_ylabel("Commits", fontsize=11)
        ax2.set_title("Commits by Hour", fontsize=12)
        ax2.set_xticks(range(0, 24, 2))

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def plot_yearly_summary(
        self,
        yearly_stats: pd.DataFrame,
        title: str = "Yearly Summary",
        save_name: str = "yearly_summary",
    ) -> str:
        """绘制年度汇总图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].bar(
            yearly_stats["year"], yearly_stats["commits"], color="#4CAF50", alpha=0.8
        )
        axes[0, 0].set_title("Commits per Year", fontsize=12)
        axes[0, 0].set_xlabel("Year")
        axes[0, 0].set_ylabel("Commits")

        axes[0, 1].bar(
            yearly_stats["year"],
            yearly_stats["unique_authors"],
            color="#2196F3",
            alpha=0.8,
        )
        axes[0, 1].set_title("Active Contributors per Year", fontsize=12)
        axes[0, 1].set_xlabel("Year")
        axes[0, 1].set_ylabel("Contributors")

        axes[1, 0].bar(
            yearly_stats["year"],
            yearly_stats["insertions"],
            color="#8BC34A",
            alpha=0.8,
            label="Insertions",
        )
        axes[1, 0].bar(
            yearly_stats["year"],
            -yearly_stats["deletions"],
            color="#F44336",
            alpha=0.8,
            label="Deletions",
        )
        axes[1, 0].set_title("Code Changes per Year", fontsize=12)
        axes[1, 0].set_xlabel("Year")
        axes[1, 0].set_ylabel("Lines")
        axes[1, 0].legend()
        axes[1, 0].axhline(y=0, color="black", linewidth=0.5)

        net_lines = yearly_stats["insertions"] - yearly_stats["deletions"]
        cumulative = net_lines.cumsum()
        axes[1, 1].fill_between(
            yearly_stats["year"], cumulative, alpha=0.3, color="#9C27B0"
        )
        axes[1, 1].plot(
            yearly_stats["year"], cumulative, color="#9C27B0", linewidth=2, marker="o"
        )
        axes[1, 1].set_title("Cumulative Net Lines", fontsize=12)
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("Lines")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def plot_code_churn(
        self,
        churn_df: pd.DataFrame,
        title: str = "Code Churn Analysis",
        save_name: str = "code_churn",
    ) -> str:
        """绘制代码搅动率分析图，展示月度代码变更总量和每次提交的平均变更量"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x = range(len(churn_df))

        ax1.fill_between(x, churn_df["churn"], alpha=0.3, color="#FF5722")
        ax1.plot(x, churn_df["churn"], color="#E64A19", linewidth=1.5)
        ax1.set_ylabel("Total Churn (lines)", fontsize=11)
        ax1.set_title("Monthly Code Churn", fontsize=12)

        ax2.fill_between(x, churn_df["churn_per_commit"], alpha=0.3, color="#3F51B5")
        ax2.plot(x, churn_df["churn_per_commit"], color="#303F9F", linewidth=1.5)
        ax2.set_ylabel("Churn per Commit", fontsize=11)
        ax2.set_xlabel("Time Period", fontsize=11)

        step = max(1, len(churn_df) // 20)
        ax2.set_xticks(list(x)[::step])
        ax2.set_xticklabels(
            churn_df["period"].iloc[::step], rotation=45, ha="right", fontsize=8
        )

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def plot_bus_factor_trend(
        self,
        bus_factor_data: List[dict],
        title: str = "Bus Factor Trend",
        save_name: str = "bus_factor_trend",
    ) -> str:
        """绘制公共汽车系数趋势图，用颜色区分风险等级"""
        df = pd.DataFrame(bus_factor_data)
        fig, ax = plt.subplots(figsize=(12, 5))

        colors = [
            "#F44336" if bf <= 2 else "#FF9800" if bf <= 3 else "#4CAF50"
            for bf in df["bus_factor"]
        ]
        ax.bar(
            df["period"],
            df["bus_factor"],
            color=colors,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.axhline(
            y=2, color="#F44336", linestyle="--", alpha=0.7, label="High Risk (≤2)"
        )
        ax.axhline(
            y=3, color="#FF9800", linestyle="--", alpha=0.7, label="Medium Risk (≤3)"
        )
        ax.set_xlabel("Period", fontsize=11)
        ax.set_ylabel("Bus Factor", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def plot_file_complexity_comparison(
        self,
        complexity_report: pd.DataFrame,
        top_n: int = 15,
        title: str = "File Complexity Comparison",
        save_name: str = "file_complexity",
    ) -> str:
        """绘制文件复杂度对比图，用颜色区分平均复杂度的高低"""
        df = complexity_report.nlargest(top_n, "lines_of_code")
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = range(len(df))
        colors = plt.cm.RdYlGn_r(
            df["avg_complexity"] / max(df["avg_complexity"].max(), 1)
        )

        ax.barh(y_pos, df["lines_of_code"], color=colors, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["file"], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Lines of Code", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")

        for i, (loc, comp) in enumerate(zip(df["lines_of_code"], df["avg_complexity"])):
            ax.text(loc + 50, i, f"Avg: {comp:.1f}", va="center", fontsize=9)

        plt.tight_layout()
        return self._save_fig(fig, save_name)


if __name__ == "__main__":
    print("可视化模块 - 请通过main.py运行")
