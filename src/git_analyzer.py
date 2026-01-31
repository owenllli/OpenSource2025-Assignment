from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
from git import Repo


class GitAnalyzer:
    """
    封装了对Git仓库的各种分析操作，并将Commit记录转为Pandas库的DataFrame方便后续处理。
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.repo = Repo(repo_path)

    def get_all_commits(self, branch: str = None) -> pd.DataFrame:
        """
        获取指定分支的所有commit记录，以及每个commit的统计信息。返回的DataFrame按时间正序排列（从最早到最新）

        Args:
            branch (str): 要分析的分支名，如果不传则使用当前活跃分支名
        """
        if branch is None:
            branch = (
                self.repo.active_branch.name
            )  # 防止分支名如master和main不同导致的问题
        commits_data = []
        for commit in self.repo.iter_commits(branch):
            stats = commit.stats.total  # 包含该次提交的统计信息
            commits_data.append(
                {
                    "hash": commit.hexsha,
                    "short_hash": commit.hexsha[:7],
                    "author_name": commit.author.name,
                    # 在cherry-pick、rebase、merge PR时，committer和author可能不同
                    "committer_name": commit.committer.name,
                    # 默认是时间戳样式，需要转换为datetime
                    "date": datetime.fromtimestamp(commit.committed_date),
                    "message": commit.message.strip(),
                    "headline": commit.message.strip().split("\n")[0],
                    "insertions": stats.get("insertions", 0),  # 新增行数
                    "deletions": stats.get("deletions", 0),  # 删除行数
                    "files_changed": stats.get("files", 0),  # 变更文件数
                }
            )
        df = pd.DataFrame(commits_data)
        df["lines_changed"] = df["insertions"] + df["deletions"]  # 总变更行数
        return df.sort_values("date").reset_index(drop=True)

    def get_commit_stats_by_author(
        self, commits_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        按作者名统计其贡献量。并计算每个作者的：
        - 提交次数
        - 新增/删除的代码行数
        - 修改的文件数
        - 首次/最后一次提交时间

        Args:
            commits_df (pd.DataFrame, optional): 提交记录。不传入则自动使用全部提交

        Returns:
            pd.DataFrame: 按作者统计的贡献表，按提交数降序排列
        """
        if commits_df is None:
            commits_df = self.get_all_commits()
        stats = (
            commits_df.groupby("author_name")
            .agg(
                {
                    "hash": "count",
                    "insertions": "sum",
                    "deletions": "sum",
                    "files_changed": "sum",
                    "date": ["min", "max"],
                }
            )
            .reset_index()
        )
        # 重命名展平后的列名
        stats.columns = [
            "author",
            "commits",
            "insertions",
            "deletions",
            "files_changed",
            "first_commit",
            "last_commit",
        ]
        stats["total_lines"] = stats["insertions"] + stats["deletions"]
        # 按提交数降序排列即贡献最多的最靠前
        return stats.sort_values("commits", ascending=False).reset_index(drop=True)

    def get_commit_stats_by_time(
        self, commits_df: pd.DataFrame = None, freq: str = "M"
    ) -> pd.DataFrame:
        """
        将提交按指定的时间频率分组，统计每个时间段内的提交情况。

        Args:
            commits_df (pd.DataFrame, optional): 提交记录
            freq (str): 时间频率，支持Pandas的时间频率字符串如Y M W D等。默认是M即按月统计。

        Returns:
            pd.DataFrame: 按时间段统计的表格。
        """
        if commits_df is None:
            commits_df = self.get_all_commits()  # 同上

        df = commits_df.copy()
        # 将日期转为时间段
        df["period"] = df["date"].dt.to_period(freq)
        stats = (
            df.groupby("period")
            .agg(
                {
                    "hash": "count",
                    "insertions": "sum",
                    "deletions": "sum",
                    "author_name": "nunique",
                }
            )
            .reset_index()
        )
        stats.columns = [
            "period",
            "commits",
            "insertions",
            "deletions",
            "unique_authors",
        ]
        stats["period"] = stats["period"].astype(str)  # 将Period类型转为字符串
        return stats

    def get_file_change_history(self, file_path: str) -> pd.DataFrame:
        """
        获取指定文件在整个仓库历史中的所有修改记录。

        Args:
            file_path (str): 文件相对路径

        Returns:
            pd.DataFrame: 该文件的修改历史，按时间正序排列。
        """
        file_commits = []
        for commit in self.repo.iter_commits(paths=file_path):
            # commit.stats.files的key是文件路径，value就是该文件在这次提交中的变更统计
            stats = commit.stats.files.get(file_path, {})
            file_commits.append(
                {
                    "hash": commit.hexsha,
                    "date": datetime.fromtimestamp(commit.committed_date),
                    "author": commit.author.name,
                    "message": commit.message.strip().split("\n")[0],
                    "insertions": stats.get("insertions", 0),
                    "deletions": stats.get("deletions", 0),
                }
            )
        df = pd.DataFrame(file_commits)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def calculate_code_churn(
        self, commits_df: pd.DataFrame = None, window: str = "M"
    ) -> pd.DataFrame:
        """
        计算代码搅动率（Code Churn）。
        代码搅动率的定义为：特定时间窗口内的代码变更总量（新增 + 删除），常用于衡量代码变更的剧烈程度。

        Args:
            commits_df (pd.DataFrame, optional): 提交记录
            window (str): 时间窗口，默认是按月即"M"

        Returns:
            pd.DataFrame: 代码搅动率统计。
        """
        if commits_df is None:
            commits_df = self.get_all_commits()
        df = commits_df.copy()
        df["period"] = df["date"].dt.to_period(window)
        churn = (
            df.groupby("period")
            .agg(
                {
                    "insertions": "sum",
                    "deletions": "sum",
                    "hash": "count",
                }
            )
            .reset_index()
        )
        churn.columns = ["period", "insertions", "deletions", "commits"]
        churn["churn"] = churn["insertions"] + churn["deletions"]  #
        churn["churn_per_commit"] = churn["churn"] / churn["commits"]
        churn["period"] = churn["period"].astype(str)  # 转字符串
        return churn

    def calculate_bus_factor(
        self, commits_df: pd.DataFrame = None, threshold: float = 0.5
    ) -> Dict:
        """
        计算公共汽车系数 (Bus Factor)。
        公共汽车系数的定义为：一个项目或项目至少失去若干关键成员的参与（典型即“被巴士撞了”）而导致项目陷入混乱、瘫痪而无法存续时，这些成员的数量。也被翻译为巴士系数。这是衡量项目风险的指标。

        计算方法：
        1. 按提交数对贡献者降序排序
        2. 计算累计贡献比例
        3. 找到累计贡献超过阈值（默认50%）所需的最少人数

        Args:
            commits_df (pd.DataFrame, optional): 提交记录
            threshold (float): 贡献占比阈值，默认为0.5即50%。

        Returns:
            Dict: 包含以下字段：
                - bus_factor: 公共汽车系数
                - threshold: 使用的阈值
                - total_authors: 总贡献者数
                - total_commits: 总提交数
                - top_contributors: 核心贡献者列表（包含姓名、提交数、占比）
        """
        if commits_df is None:
            commits_df = self.get_all_commits()
        author_stats = self.get_commit_stats_by_author(commits_df)
        total_commits = author_stats["commits"].sum()

        # 每个作者的提交占比
        author_stats["commit_ratio"] = author_stats["commits"] / total_commits

        # 计算累计占比（从贡献最多的开始累加）
        author_stats["cumulative_ratio"] = author_stats["commit_ratio"].cumsum()

        bus_factor = (author_stats["cumulative_ratio"] <= threshold).sum() + 1
        bus_factor = min(bus_factor, len(author_stats))  # 防止阈值设置成100%时溢出

        top_contributors = author_stats.head(bus_factor)[
            ["author", "commits", "commit_ratio"]
        ].to_dict("records")

        return {
            "bus_factor": bus_factor,
            "threshold": threshold,
            "total_authors": len(author_stats),
            "total_commits": total_commits,
            "top_contributors": top_contributors,
        }

    def get_commit_time_distribution(
        self, commits_df: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分析开发者的提交时间分布。统计提交在一周中各天和一天中各小时的分布情况。

        Args:
            commits_df (pd.DataFrame, optional): 提交记录

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 返回两个表格
            weekday_stats: 按星期统计
                - weekday: 星期几（Monday, Tuesday, ...）
                - commits: 该天的提交次数
            hour_stats: 按小时统计
                - hour: 小时（0-23）
                - commits: 该小时的提交次数
        """
        if commits_df is None:
            commits_df = self.get_all_commits()

        df = commits_df.copy()

        df["weekday"] = df["date"].dt.day_name()
        df["hour"] = df["date"].dt.hour  # 返回 0-23

        # 定义星期顺序（确保输出按周一到周日排列）
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        # 按星期统计
        weekday_stats = (
            df.groupby("weekday")["hash"]
            .count()
            .reindex(weekday_order)  # 按指定顺序重排
            .reset_index()
        )
        weekday_stats.columns = ["weekday", "commits"]

        # 按小时统计
        hour_stats = df.groupby("hour")["hash"].count().reset_index()
        hour_stats.columns = ["hour", "commits"]

        return weekday_stats, hour_stats

    def get_yearly_summary(self, commits_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        生成年度汇总统计。按年份统计项目的发展情况，包括该年的提交次数、贡献者数量等信息。

        Args:
            commits_df (pd.DataFrame, optional): 提交记录

        Returns:
            pd.DataFrame: 年度汇总表
        """
        if commits_df is None:
            commits_df = self.get_all_commits()

        df = commits_df.copy()
        df["year"] = df["date"].dt.year

        summary = (
            df.groupby("year")
            .agg(
                {
                    "hash": "count",
                    "author_name": "nunique",
                    "insertions": "sum",
                    "deletions": "sum",
                }
            )
            .reset_index()
        )
        summary.columns = [
            "year",
            "commits",
            "unique_authors",
            "insertions",
            "deletions",
        ]

        summary["total_lines"] = summary["insertions"] + summary["deletions"]

        return summary


def main():
    import sys

    if len(sys.argv) >= 2:
        repo_path = sys.argv[1]
    else:  # 方便直接运行
        repo_path = "../tqdm"

    analyzer = GitAnalyzer(repo_path)
    commits = analyzer.get_all_commits()
    print(f"\n共获取 {len(commits)} 条提交记录")
    print(
        f"  时间范围: {commits['date'].min().strftime('%Y-%m-%d')} ~ {commits['date'].max().strftime('%Y-%m-%d')}"
    )

    author_stats = analyzer.get_commit_stats_by_author(commits)
    print(author_stats.head(10).to_string(index=False))

    bus = analyzer.calculate_bus_factor(commits)
    print(
        f"\n公共汽车系数: {bus['bus_factor']}(前 {bus['bus_factor']} 名贡献者贡献了 50% 以上的提交)"
    )

    print("\n年度统计:")
    yearly = analyzer.get_yearly_summary(commits)
    print(yearly.to_string(index=False))


if __name__ == "__main__":
    main()
