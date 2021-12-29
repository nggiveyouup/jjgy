<!-- @import "ACM.less" -->

# ACM Cheat Table

> ~~Author:~~ *Anonymous*
> Version: v0.1.0
> Date Created: 2021-12-28
> Date Completed: 2021-12-29
> **Contact the administrators on the copyright of this article.**

<!-- Reference: https://blog.csdn.net/xyqqwer/article/details/81433429 -->
<!-- Reference: https://zhuanlan.zhihu.com/p/122413160 -->

## I. 算法基础

### I.I. 埃拉托斯特尼筛法快速筛选质数

```c++
int prime[maxn];  
bool is_prime[maxn];
 
int sieve(int n){
    int p = 0;
    for(int i = 0; i <= n; ++i)
        is_prime[i] = true;
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i <= n; ++i){  // 注意数组大小是n
        if(is_prime[i]){
            prime[p++] = i;
            for(int j = i + i; j <= n; j += i)  // 轻剪枝，j必定是i的倍数
                is_prime[j] = false;
        }
    }
    return p;  // 返回素数个数
}
```

### I.II. 快速幂

```c++
typedef long long LL;

// res = x ** n % m
LL powerMod(LL x, LL n, LL m) {
    LL res = 1;
    while (n > 0) {
        // n是奇数取true
        if (n & 1) res = (res * x) % m;
        x = (x * x) % m;
        n >>= 1;  // n /= 2
    }
    return res;
}
```

### I.III. 高精度运算

#### I.III.I. 加法

```c++
string add1(string s1, string s2)
{
    if (s1 == "" && s2 == "")   return "0";
    if (s1 == "")   return s2;
    if (s2 == "")   return s1;
    string maxx = s1, minn = s2;
    if (s1.length() < s2.length()){
        maxx = s2;
        minn = s1;
    }
    int a = maxx.length() - 1, b = minn.length() - 1;
    for (int i = b; i >= 0; --i){
        maxx[a--] += minn[i] - '0'; //  a一直在减 ， 额外还要减个'0'
    }
    for (int i = maxx.length()-1; i > 0;--i){
        if (maxx[i] > '9'){
            maxx[i] -= 10;//注意这个是减10
            maxx[i - 1]++;
        }
    }
    if (maxx[0] > '9'){
        maxx[0] -= 10;
        maxx = '1' + maxx;
    }
    return maxx;
}
```

#### I.III.II. 阶乘

```c++
typedef long long LL;
 
const int maxn = 100010;
 
int num[maxn], len;
 
/*
    在mult函数中，形参部分：len每次调用函数都会发生改变，n表示每次要乘以的数，最终返回的是结果的长度
    tip: 阶乘都是先求之前的(n-1)!来求n!
    初始化Init函数很重要，不要落下
*/
 
void Init() {
    len = 1;
    num[0] = 1;
}
 
int mult(int num[], int len, int n) {
    LL tmp = 0;
    for(LL i = 0; i < len; ++i) {
         tmp = tmp + num[i] * n;  // 从最低位开始，等号左边的tmp表示当前位，右边的tmp表示进位（之前进的位）
         num[i] = tmp % 10;  // 保存在对应的数组位置，即去掉进位后的一位数
         tmp = tmp / 10;  // 取整用于再次循环,与n和下一个位置的乘积相加
    }
    while(tmp) {  // 之后的进位处理
         num[len++] = tmp % 10;
         tmp = tmp / 10;
    }
    return len;
}
 
int main() {
    Init();
    int n;
    n = 1977;  // 求的阶乘数
    for(int i = 2; i <= n; ++i) {
        len = mult(num, len, i);
    }
    for(int i = len - 1; i >= 0; --i)
        printf("%d",num[i]);    //  从最高位依次输出,数据比较多采用printf输出
    printf("\n");
    return 0;
}
```

### I.IV. 辗转相除法

#### I.IV.I. GCD

```c++
int gcd(int big, int small)
{
    if (small > big) swap(big, small);
    int temp;
    while (small != 0){  // 辗转相除法
        if (small > big) swap(big, small);
        temp = big % small;
        big = small;
        small = temp;
    }
    return big;
}
```

#### I.IV.II. LCM

```c++
lcm = big * small / gcd(big, small);
```

### I.V. 排列组合

#### I.V.I. 排列

##### I.V.I.I. 可重复

```c++
// 默认是升序
// isValid = false: 下一个排列不存在
bool prev_permutation(iterator start, iterator end, BinaryPredicate cmp);
bool next_permutation(iterator start, iterator end, BinaryPredicate cmp);
bool isValid = next_permutation(list, list + n);
bool cmp(int a, int b) {return a < b;}
```

##### I.V.I.II. 不可重复

```c++
class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> track;  // 路径：在决策树中已经做出的选择
        back_track(n,k,1,track,res); 
        return res;
    }

    void back_track(int n,int k,int start,vector<int> &track,vector<vector<int>> &res){
        if(track.size()==k){  // 递归结束条件：路径大小==k，到达决策树底层
            res.push_back(track);
            return;
        }
        for(int i=start;i<=n;++i){  // i 从 start 开始递增
            track.push_back(i);
            back_track(n,k,i+1,track,res);
            track.pop_back();
        }
    }
};
```

#### I.V.II. 组合

```c++
class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> track;   // 路径：在决策树中已经做出的选择
        back_track(n,k,1,track,res); 
        return res;
    }

    void back_track(int n,int k,int start,vector<int> &track,vector<vector<int>> &res){
        if(track.size()==k){   // 递归结束条件：路径大小==k，到达决策树底层
            res.push_back(track);
            return;
        }
        for(int i=start;i<=n;++i){   // i 从 start 开始递增
            track.push_back(i);
            back_track(n,k,i+1,track,res);
            track.pop_back();
        }
    }
};
```

### I.VI. 子集

#### I.VI.I. 可重复

```c++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> track;
        back_track(nums,0,track,res);
        return res;
    }

    void back_track(vector<int>& nums,int start,vector<int>& track,vector<vector<int>>& res){
        res.push_back(track);
        int siz=nums.size();
        for(int i=start;i<siz;++i){
            track.push_back(nums[i]);
            back_track(nums,i+1,track,res);
            track.pop_back();
        }
    }
};
```

#### I.VI.II. 不可重复

```c++
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> track;
        sort(nums.begin(),nums.end());
        back_track(nums,0,track,res);
        return res;
    }

    void back_track(vector<int>& nums,int start,vector<int> &track,vector<vector<int>> &res){  
        res.push_back(track);     
        int siz=nums.size();
        for(int i=start;i<siz;++i){
            if(i>start && nums[i]==nums[i-1]){
                continue;
            }
            track.push_back(nums[i]);
            back_track(nums,i+1,track,res);
            track.pop_back();
        }
    }
};
```

### I.VII. 二分查找

```c++
bool isFound = binary_search(arr, arr + n, target, cmp);
// lower_bound: 查找第一个大于或等于某个元素的位置，返回迭代器、指针等
int* ptr = lower_bound(arr, arr + n, target, cmp);
// upper_bound: 查找第一个大于某个元素的位置，返回迭代器、指针等
int* ptr = upper_bound(arr, arr + n, target, cmp);
// lower_bound(arr, arr + n, x): 最后一个x下一个元素的地址
// upper_bound(arr, arr + n, x): 第一个x的地址
// upper_bound(arr, arr + n, x) - lower_bound(arr, arr + n, x): x的个数
// lower_bound(arr, arr + n, x) - arr: 下标
```

```c++
int binarySearch(int x,int n)
{
    int left =0;
    int right=n-1;
    while(left<=right)
    {
        int mid =(left+right)/2;
        if(x==a[mid]) return mid;
        if(x>a[mid]) left=mid+1; else right =mid-1;
    }
    return -1;  // 未找到x
}
```

### I.VIII. 倍增

用1、2、4、...、2^(m-1)表示[1,2^m)之间的所有数。

例：区间最值查询（RMQ）（预处理O(nlog(n))，查询O(1)）

<!-- Reference: https://blog.csdn.net/MikeJackSTG/article/details/81806120 -->

```c++
// f[i][j]: [i, i+2^j)的最大值
// 预处理
void ST(int n) {
    for (int i = 1; i <= n; ++i) {
        f[i][0] = a[i];
    }
    for (int j = 1; (1 << j) <= n; ++j) {
        for (int i = 1; i + (1 << j) <= n + 1; ++j) {
            f[i][j] = max(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);
        }
    }
}

// 查询[l, r)最值
int RMQ(int l, int r) {
    int k = trunc(log2(r - l));
    return max(f[l][k], f[r - (1 << k)][k]);
}
```

## II. 图论算法

1. **图：** 若干点和若干边
2. **简单图：** 无平行边（有向图中还要求方向相同）和自环
3. **稀疏图：** 边数远小于完全图边数
4. **稠密图：** 边数接近或等于完全图边数
5. **连通图：** 无向图，任意两点相通
6. **强连通图：** 有向图，任意两点相通
7. **连通网：** 加权连通图
8. **生成树：** 连通子图，n个节点，n-1条边
9. **最小生成树：** 代价最小的生成树

### II.I. 最小生成树（MST）

#### II.I.I. Kruskal算法（简单图，并查集，加边法）

```c++
/*
edge.a, edge.b: 左右节点
edge.len: 权值
*/

void Kruskal() {
    ans = 0, k = 0;  // k: 已经连接的边数
    sort(edge, edge + len, cmp);  // 升序
    for (int i = 0; /* i < len && */ k < len - 1; ++i) {
        if (Find(edge[i].a) != Find(edge[i].b)) {
            Union(edge[i].a, edge[i].b);
            ans += edge[i].len;
            ++k;
        }
    }
}
```

#### II.I.II. Prim算法（稠密图，优先队列，O(elog(n))）

```c++
struct edge  // 保存边的情况，to为通往的边，不需要保存from
{
    int to;
    int v;
    friend bool operator<(const edge& x,const edge& y)  // 优先队列即最小堆
    {
        return x.v>y.v;
    }
};

priority_queue<edge> q;
int vis[105] = {0};  // 判断是否标记数组
int p[105][105];  // 存图

void Prim() {
    s = 0;  // 答案
    vis[1] = 1;
    key = 1;  // 起点
    edge now;
    while (!q.empty()) q.pop();  // 清空
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j <= n; ++j) {  // 新点
            if (!vis[j]) {  // 没到过的点
                now.to = j;
                now.v = p[key][j];
                q.push(now);
            }
        }
        while (!q.empty() && vis[q.top().to]) q.pop();  // 最小边但到过，忽略
        if (q.empty()) break;
        now = q.top();
        key = now.to;
        s += now.v;
        vis[key] = 1;
        q.pop();
    }
    return s;
}
```

### II.II. 最短路径

#### II.II.I. Dijkstra算法（单源最短路，正边权，有向图和无向图，O(elog(n))）

```c++

struct node {  
    int v, len;  
    node(int v = 0, int len = 0) :v(v), len(len) {}  
    bool operator < (const node &a)const {  //  距离从小到大排序  
        return len > a.len;  
    }  
};  
 
vector<node>G[maxn];  
bool vis[maxn];  
int dis[maxn];
 
void init() {  
    for (int i = 0; i<maxn; i++) {  
        G[i].clear();  
        vis[i] = false;  
        dis[i] = INF;  
    }  
}  
int dijkstra(int s, int e) {  
    priority_queue<node>Q;  
    Q.push(node(s, 0)); //  加入队列并排序  
    dis[s] = 0;  
    while (!Q.empty()) {  
        node now = Q.top();     //  取出当前最小的  
        Q.pop();  
        int v = now.v;  
        if (vis[v]) continue;   //  如果标记过了, 直接continue  
        vis[v] = true;  
        for (int i = 0; i<G[v].size(); i++) {   //  更新  
            int v2 = G[v][i].v;  
            int len = G[v][i].len;  
            if (!vis[v2] && dis[v2] > dis[v] + len) {  
                dis[v2] = dis[v] + len;  
                Q.push(node(v2, dis[v2]));  
            }  
        }  
    }  
    return dis[e];  
}
```

#### II.II.II. SPFA最短路径快速算法（队列，负环）

```c++

vector<node> G[maxn];
bool inqueue[maxn];
int dist[maxn];
 
void Init()  
{  
    for(int i = 0 ; i < maxn ; ++i){  
        G[i].clear();  
        dist[i] = INF;  
    }  
}  
int SPFA(int s,int e)  
{  
    int v1,v2,weight;  
    queue<int> Q;  
    memset(inqueue,false,sizeof(inqueue)); // 标记是否在队列中  
    memset(cnt,0,sizeof(cnt)); // 加入队列的次数  
    dist[s] = 0;  
    Q.push(s); // 起点加入队列  
    inqueue[s] = true; // 标记  
    while(!Q.empty()){  
        v1 = Q.front();  
        Q.pop();  
        inqueue[v1] = false; // 取消标记  
        for(int i = 0 ; i < G[v1].size() ; ++i){ // 搜索v1的链表  
            v2 = G[v1][i].vex;  
            weight = G[v1][i].weight;  
            if(dist[v2] > dist[v1] + weight){ // 松弛操作  
                dist[v2] = dist[v1] + weight;  
                if(inqueue[v2] == false){  // 再次加入队列  
                    inqueue[v2] = true;  
                    //cnt[v2]++;  // 判负环  
                    //if(cnt[v2] > n) return -1;  
                    Q.push(v2);  
                } } }  
    }  
    return dist[e];  
}
 
/*
    不断的将s的邻接点加入队列，取出不断地进行松弛操作，直到队列为空  
    如果一个结点被加入队列超过n-1次，那么显然图中有负环  
*/
```

#### II.II.III. Floyd-Warshall算法（任意点对）

```c++
for (int i = 0; i < n; i++) {   //  初始化为0  
    for (int j = 0; j < n; j++)  
        scanf("%lf", &dis[i][j]);  
}  
for (int k = 0; k < n; k++) {  
    for (int i = 0; i < n; i++) {  
        for (int j = 0; j < n; j++) {  
            dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j]);  
        }  
    }
}
```

## III. 动态规划

### III.I. 背包问题

<!-- Reference: https://zhuanlan.zhihu.com/p/93857890 -->

1. **01背包：** N件物品，重量w[i]，价值v[i]，容量W
2. **完全背包：** N种物品，重量w[i]，价值v[i]，容量W
3. **多重背包：** N种物品，数量n[i]，重量w[i]，价值v[i]，容量W

#### III.I.I. 01背包

```c++
if (j >= w[i]) {
    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i]);
}
```

```c++
for (int i = 1; i <= N; ++i) {
    for (int j = W; j >= w[i]; --j) {
        dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    }
}
```

#### III.I.II. 完全背包

```c++
if (j >= w[i]) {
    // 数量无限，装完还能继续装，所以第二项第一维下标是i不是i-1
    dp[i][j] = max(dp[i - 1][j], dp[i][j - w[i]] + v[i]);
}
```

```c++
for (int i = 1; i <= N; ++i) {
    for (int j = w[i]; j <= W; ++j) {
        dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    }
}
```

```c++
for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= W; ++j) {
        for (int k = 0; k <= j / w[i]) {
            if (j >= k * w[i]) {
                dp[i][k] = max(dp[i][k], dp[i - 1][j - k * w[i]] + k * v[i]);
            }
        }
    }
}
```

#### III.I.III. 多重背包

```c++
for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= W; ++j) {
        for (int k = 1; k <= min(n[i], j / w[i])) {
            if (j >= k * w[i]) {
                dp[i][k] = max(dp[i][k], dp[i - 1][j - k * w[i]] + k * v[i]);
            }
        }
    }
}
```

```c++
for (int i = 1; i <= N; ++i) {
    for (int j = W; j >= w[i]; --j) {
        for (int k = 0; k <= min(n[i], j / w[i])) {
            dp[j] = max(dp[j], dp[j - k * w[i]] + k * v[i]);
        }
    }
}
```

#### III.I.IV. 变式

1. **恰好装满：** `dp[][0] = 0; dp[][>0] = -inf;`
2. **方案总数：** 状态转移方程中`max`改为求和
3. **二维背包：** 增加`dp`数组维数
4. **具体取法：** 用新的数组存储状态转移方程中具体取了哪一项

### III.II. 子序列问题

#### III.II.I. 最长上升子序列（LIS）

```c++
// O(nlog(n))，num原串，lis用以dp的数组，pos原串在lis中出现的位置，ans所求LIS
int last = 0;  // LIS长度
pos[0] = 1;
for (int i = 1; i <= n; ++i) {
    if (num[i] > lis[last]) {
        lis[++last] = num[i];
        pos[i] = last;
    };
    else {
        // 可以不用lis + last + 1，因为肯定不大于末尾元素
        int k = lower_bound(lis + 1, lis + last, num[i]) - lis;
        lis[k] = num[i];
        pos[i] = k;
    }
}
for (int i = n - 1; i >= 0; --i) {
    if (pos[i] == last) {
        ans[last--] = i;
    }
}
```

Dilworth定理：最长上升子序列个数 = 最长不上升子序列长度，最长下降子序列个数 = 最长不下降子序列长度。

#### III.II.II. 最长公共子序列（LCS）

```c++
if (a[i] == b[j]) {
    dp[i][j] = dp[i - 1][j - 1] + 1;
} else {
    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
}
```

#### III.II.III. 最长回文子序列

```text
LCS(S, S的反转串)
```

### III.III. 子串问题

注意出现间断则立即将dp更新为0。

例：最长公共子串

```c++
if (a[i] == b[j]) {
    dp[i][j] = dp[i - 1][j - 1] + 1;
} else {
    dp[i][j] = 0;
}
```

## IV. 字符串问题

### IV.I. KMP算法

<!-- Reference: https://zhuanlan.zhihu.com/p/83334559 -->

```c++
// 二维dp数组：
// 状态只与pat有关
int dp[MAXM][256];  // dp[状态][字符] = 下个状态，数值表示已经匹配完成的字符串长度

void KMP(string pat) {
    int M = pat.length();
    dp[0][pat[0]] = 1;  // 初态
    int X = 0;  // 影子状态，与当前状态的字符串前缀相同
    for (int j = 1; j < M; ++j) {
        // 未匹配上，下一状态由影子状态决定（匹配上的情况被重复计算）
        for (int c = 0; c < 256; ++c) {
            // 影子状态永远不会超过当前状态，而之前的状态对应的结果已经全部被计算出
            dp[j][c] = dp[X][c];
        }
        // 匹配上，状态推进
        dp[j][pat[j]] = j + 1;
        X = dp[X][pat[j]];
    }
}

int Search(string txt) {
    int M = pat.length(), N = txt.length();
    int j = 0;  // pat初态
    for (int i = 0; i < N; ++i) {
        j = dp[j][txt[i]];  // 计算下一状态
        if (j == M) return i - M + 1;  // 到达最终状态（匹配完毕）
    }
    return -1;
}
```

```c++
// 一维dp（next）数组：
int dp[MAXM];

void KMP(string pat) {
    int j = 0, k = -1;
    dp[0] = -1;
    while (j < pat.length() - 1) {
        if (k == -1 || t[j] == t[k]) {
            ++j; ++k;
            // 字符相同，跳过
            if (pat[j] == pat[k]) {
                dp[j] = dp[k];
            } else {
                dp[j] = k;
            }
        } else k = next[k];
    }
}

int KMP(string pat, string txt) {
    int i = 0, j = 0;
    while (i < txt.length() && j < pat.length()) {
        if (j == -1 || txt[i] == pat[j]) {
            ++i; ++j;
        } else j = dp[j];
    }
    if (j >= pat.length()) {
        return i - pat.length();
    } else {
        return -1;
    }
}
```

### IV.II. 字典树（Trie）

### IV.III. AC自动机（多模匹配）

## V. 常用数据结构、类和STL

### IV.I. String

### IV.I. Iterator

### IV.II. Vector

### IV.III. Pair

### IV.IV. Priority Queue

### IV.V. Map

### IV.VI. 并查集

### V.VII. 线段树

思想类似倍增。解决满足区间可加性的点修改、区间查询、区间修改。