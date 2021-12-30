<!-- @import "ACM.less" -->

# ACM Cheat Table

> ~~Author:~~ *Anonymous*
> Version: v1.0.0
> Date Created: 2021-12-29
> Date Completed: 2021-12-30
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

思想类似倍增。

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

```c++
struct Trie {
    int cnt;
    Trie *next[maxn];
    Trie() {
        cnt = 0;
        memset(next, 0, sizeof(next));
    }
};

Trie *root;

void Insert(char *word) {
    Trie *tem = root;
    while (*word != '\0') {
        int x = *word - 'a';
        if (tem->next[x] == NULL) tem->next[x] = new Trie;
        tem = tem->next[x];
        tem->cnt++;
        word++;
    }
}

int Search(char *word) {
    Trie *tem = root;
    for (int i = 0; word[i] != '\0'; i++) {
        int x = word[i] - 'a';
        if (tem->next[x] == NULL) return 0;
        tem = tem->next[x];
    }
    return tem->cnt;
}

void Delete(char *word, int t) {
    Trie *tem = root;
    for (int i = 0; word[i] != '\0'; i++) {
        int x = word[i] - 'a';
        tem = tem->next[x];
        (tem->cnt) -= t;
    }
    for (int i = 0; i < maxn; i++) tem->next[i] = NULL;
}

int main() {
    int n;
    char str1[50];
    char str2[50];
    while (scanf("%d", &n) != EOF) {
        root = new Trie;
        while (n--) {
            scanf("%s %s", str1, str2);
            if (str1[0] == 'i') Insert(str2);
            else if (str1[0] == 's') {
                if (Search(str2)) printf("Yes\n");
                else printf("No\n");
            } else {
                int t = Search(str2);
                if (t) Delete(str2, t);
            }
        }
    }
    return 0;
}
```

### IV.III. AC自动机（多模匹配）

```c++
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>

using namespace std;

#define N 1000010

char str[N], keyword[N];
int head, tail;

struct node {
    node *fail;
    node *next[26];
    int count;
    node() {  // init
        fail = NULL;  // 默认为空
        count = 0;
        for (int i = 0; i < 26; ++i) next[i] = NULL;
    }
} * q[N];

node *root;

void insert(char *str) {  // 建立Trie
    int temp, len;
    node *p = root;
    len = strlen(str);
    for (int i = 0; i < len; ++i) {
        temp = str[i] - 'a';
        if (p->next[temp] == NULL) p->next[temp] = new node();
        p = p->next[temp];
    }
    p->count++;
}

void build_ac() {  // 初始化fail指针，BFS 数组模拟队列：
    q[tail++] = root;
    while (head != tail) {
        node *p = q[head++]; // 弹出队头
        node *temp = NULL;
        for (int i = 0; i < 26; ++i) {
            if (p->next[i] != NULL) {
                if (p == root) // 第一个元素fail必指向根
                    p->next[i]->fail = root;
                else {
                    temp = p->fail; // 失败指针
                    while (temp != NULL) {  // 2种情况结束：匹配为空or找到匹配
                        if (temp->next[i] != NULL) {  // 找到匹配
                            p->next[i]->fail = temp->next[i];
                            break;
                        }
                        temp = temp->fail;
                    }
                    if (temp == NULL) // 为空则从头匹配
                        p->next[i]->fail = root;
                }
                q[tail++] = p->next[i]; // 入队
            }
        }
    }
}

int query() // 扫描
{
    int index, len, result;
    node *p = root; // Tire入口
    result = 0;
    len = strlen(str);
    for (int i = 0; i < len; ++i) {
        index = str[i] - 'a';
        while (p->next[index] == NULL && p != root) // 跳转失败指针
            p = p->fail;
        p = p->next[index];
        if (p == NULL) p = root;
        node *temp = p; // p不动，temp计算后缀串
        while (temp != root && temp->count != -1) {
            result += temp->count;
            temp->count = -1;
            temp = temp->fail;
        }
    }
    return result;
}

int main() {
    int num;
    head = tail = 0;
    root = new node();
    scanf("%d", &num);
    getchar();
    for (int i = 0; i < num; ++i) {
        scanf("%s", keyword);
        insert(keyword);
    }
    build_ac();
    scanf("%s", str);
    if (query()) printf("YES\n");
    else printf("NO\n");
    return 0;
}

/*
    假设有N个模式串，平均长度为L；文章长度为M。 建立Trie树：O(N*L) 建立fail指针：O(N*L) 模式匹配：O(M*L) 所以，总时间复杂度为:O( (N+M)*L )。
*/
```

## V. 常用数据结构、类和STL

<!-- Reference: http://c.biancheng.net/stl/ -->

### V.I. String

**1. 构造**

```c++
string str;
string str = "ABC";
string str("ABC");
string str("ABC", maxlen);
string str("ABC", startpos, strlen);
string str(repeattimes, 'A');
```

**2. 常用成员函数**

```c++
str.assign("ABC", 1, 2);        // 清空重新赋值，从某下标开始，保留指定长度
int len = str.length();         // 获取长度
str.resize(newlen);             // 设置长度，不足补'\0'
str.resize(newlen, fillchar);   // 设置长度，不足补fillchar
str.swap(anotherstr);           // 交换值
str.push_back('A');             // 末尾添加字符
str.append("ABC");              // 末尾添加字符串
str.insert(2, "ABC");           // 插入字符
str.erase(2);                   // 清除指定下标及之后的所有值
str.erase(2, 1);                // 清除指定下标及之后的指定长度的值
str.clear();                    // 清空
str.replace(2, 4, "ABCD");      // 替换指定下标及之后的指定长度的值
bool isempty = str.empty();     // 判断是否为空
```

**3. 反转**

```c++
string str("ABC");
reverse(str.begin(), str.end());  // 反转[first, last)范围内的元素的顺序
cout << str << endl;
```

**4. 查找**

```c++
string str("ABCDEFGABCD");                      //11个字符
int n;

/*查找成功返回位置,查找失败,则n等于-1*/
/*find():从头查找某个字符串*/
n= str.find('A');              //查找"A",n=0;
n= str.find("AB");             //查找"AB",n=0;
n= str.find("BC",1);           //从位置1处,查找"BC",n=1;
n= str.find("CDEfg",1,3);      //从位置1处,查找"CDEfg"的前3个字符,等价于str.find("CDE",1),n=2;

/*rfind():反向(reverse)查找,从末尾处开始,向前查找*/
n= str.rfind("CD");           //从位置10开始向前查找,n=9
n= str.rfind("CD",5);         //从位置5开始向前查找,n=2
n= str.rfind("CDEfg",5,3);    //等价于str.rfind("CDE",5);       ,所以n=2

/* find_first_of ():查找str里是否包含有子串中任何一个字符*/
n= str.find_first_of("abcDefg");     //由于str位置3是'D',等于"abcDefg"的'D',所以n=3
n= str.find_first_of("abcDefg",1,4); //等价于str. find_first_of ("abcD",1); 所以n=3

/* find_last_of ():末尾查找, 从末尾处开始,向前查找是否包含有子串中任何一个字符*/
n= str.find_last_of("abcDefg");      //由于str末尾位置10是'D',所以n=10
n= str.find_last_of("abcDefg",5,4);  //等价于str. find_last_of ("abcD",5); 所以n=3

/* find_first_not_of ():匹配子串任何一个字符,若某个字符不相等则返回str处的位置,全相等返回-1*/
n= str.find_last_not_of("ABC");    //由于str位置3'D',在子串里没有,所以 n=3
n= str.find_last_not_of("aABDC");  //由于str位置4 'F',在子串里没有,所以 n=4
n= str.find_last_not_of("aBDC");   //由于str位置0 'A',在子串里没有,所以 n=0

/* find_last_not_of ():反向匹配子串任何一个字符,若某个字符不相等则返回str处的位置,全相等返回-1*/
n= str.find_last_not_of("aBDC");  //由于str位置7'A',在子串里没有,所以 n=7
```

**5. 复制和子串**

```c++
str2=str1.substr(2);        //提取子串,提取出str1的下标为2到末尾,给str2
str2=str1.substr(2,3);     //提取子串,从 str1的下标为2开始,提取3个字节给str2
const char *s1= str.data();   //将string类转为字符串数组,返回给s1
char *s=new char[10];
str.copy(s,count,pos);    //将str里的pos位置开始,拷贝count个字符,存到s里.
```

### V.II. Iterator

**1. 构造**

```c++
Class::iterator iter;               // 正向迭代器
Class::const_iterator iter;         // 常量正向迭代器
Class::reverse_iterator iter;       // 反向迭代器
Class::const_reverse_iterator iter; // 常量反向迭代器
```

**2. 迭代**

```c++
vector<int> v{0, 1, 2, 3, 4};

vector<int>::iterator iter;
assert(*(++iter) == 1);  // 下一个元素
vector<int>::reverse_iterator iter;
assert(*(++iter) == 3);  // 上一个元素

for (vector<int>::iterator i = v.begin(); i != v.end(); ++i);
for (vector<int>::reverse_iterator i = v.rbegin(); i != v.rend(); ++i);
```

**3. 分类**

1. **正向迭代器：** `++p`, `p++`, `*p`, `iter1 = iter2`, `==`, `!=`
2. **双向迭代器：** 正向迭代器, `--p`, `p--`
3. **随机访问迭代器：** 双向迭代器, `+=`, `-=`, `iter + integer`, `iter - integer`, `iter[integer]`, `<`, `>`, `<=`, `>=`, `iter1 - iter2`

| 容器 | 迭代器种类 |
| :-: | :-: |
| 数组、vector、deque | 随机访问 |
| list、set、multiset、map、multimap | 双向 |
| stack、queue、priority_queue | 不支持 |

**4. 成员函数**

```c++
advance(iter, n);           // 前后移动n个元素
distance(iter1, iter2);     // 计算距离（前者不大于后者）
iter_swap(iter1, iter2);    // 交换指向的值
```

### V.III. Pair（键值对）

**1. 构造**

```c++
pair<string, double> pair1;
pair1.first = "key";
pair1.second = 0;
pair<string, string> pair2("first", "second");
pair<string, string> pair3(pair2);
pair<string, string> pair4(make_pair("first", "second"));
```

**2. 比较**

先比first成员，再比second成员。

**3. 成员函数**

```c++
pair1.swap(pair2);
```

### V.IV. Vector和Double-ended Queue

vector可以尾端操作花费常量时间，deque可以两端操作花费常量时间（数据未必存储在连续空间上）。

**1. 构造**

```c++
vector<int> vec;
vector<int> vec{0, 1, 2, 3};
vector<int> vec(10, 1);  // 指定长度，初始化为1
vector<int> vec(vec1);
vector<int> vec(begin(vec1), end(vec1));

deque<int> que;
deque<int> que{0, 1, 2, 3};
deque<int> que(10, 1);  // 指定长度，初始化为1
deque<int> que(que1);
deque<int> que(begin(que1), end(que1));
```

**2. 成员函数**

```c++
vec.begin();
vec.end();
vec.rbegin();
vec.rend();
vec.size();
vec.resize();
vec.empty();
vec.front();        // 首端元素引用
vec.back();         // 尾端元素引用
vec.data();         // 首端元素指针
vec.assign();
vec.push_back();
vec.emplace_back(); // 原地尾端插入
vec.pop_back();
vec.insert();
vec.emplace();      // 原地指定位置插入
vec.erase();
vec.clear();
vec.swap();

que.begin();
que.end();
que.rbegin();
que.rend();
que.size();
que.resize();
que.empty();
que.front();            // 首端元素引用
que.back();             // 尾端元素引用
que.assign();
que.push_front();
que.emplace_front();    // 原地首端插入
que.push_back();
que.emplace_back();     // 原地尾端插入
que.pop_front();
que.pop_back();
que.insert();
que.emplace();          // 原地指定位置插入
que.erase();
que.clear();
que.swap();
```

### V.V. Priority Queue

priority_queue：优先队列、堆。

定义：

```c++
template <typename T,                       // 元素类型
        typename Container=std::vector<T>,  // 底层容器类型，只能是vector或deque
        typename Compare=std::less<T> >     // less（从大到小）、greater（从小到大）、自定义函数
class priority_queue {}
```

**1. 构造**

```c++
int values[]{4, 1, 3, 2};
priority_queue<int> pque;
priority_queue<int> pque(values, values + 4);  // {4, 2, 3, 1}
priority_queue<int> pque(begin(values), end(values), deque<int>, greater<int>); // {1, 3, 2, 4}
```

**2. 成员函数**

```c++
pque.size();
pque.empty();
pque.top();                     // 首端元素引用
pque.push(const T& obj);        // 压入对象副本
pque.push(T&& obj);             // 压入对象本身
pque.emplace(Args&&... args);
pque.pop();
pque.swap(pque1);
```

**3. 自定义比较**

<!-- Reference: https://www.cnblogs.com/yalphait/articles/8889221.html -->

```c++
// 重载运算符
bool operator < (T a, T b) {
    // true表示a优先级低于b
}

// 自定义比较函数
struct cmp {
    bool operator() (T a, T b) {
        // true表示a优先级低于b
    }
};
priority_queue<T, vector<T>, cmp> pque;
```

### V.VI. List（双向链表）

**1. 构造**

```c++
list<int> lst;
list<int> lst(10);
list<int> lst(10, 1);
list<int> lst(values, values + 4);
list<int> lst(begin(values), end(values));
```

**2. 成员函数**

```c++
lst.begin();
lst.end();
lst.rbegin();
lst.rend();
lst.size();
lst.resize();
lst.empty();
lst.front();            // 首端元素引用
lst.back();             // 尾端元素引用
lst.assign();
lst.push_front();
lst.emplace_front();    // 原地首端插入
lst.push_back();
lst.emplace_back();     // 原地尾端插入
lst.pop_front();
lst.pop_back();
lst.insert();
lst.emplace();          // 原地指定位置插入
lst.erase();
lst.clear();
lst.swap();
lst.remove(val);        // 删除值为val的元素
lst.remove_if();        // 删除符合条件的元素
lst.unique();           // 删除容器中相邻重复元素，保留其一
lst.merge();            // 合并两个有序容器
lst.sort();
lst.reverse();          // 容器内元素反转顺序
```

### V.VII. Unordered Map（哈希表）

定义：

```c++
template <class Key,                        //键值对中键的类型
        class T,                            //键值对中值的类型
        class Hash=hash<Key>,               //容器内部存储键值对所用的哈希函数
        class Pred=equal_to<Key>,           //判断各个键值对键相同的规则
        class Alloc=allocator<pair<const Key, T> >  // 指定分配器对象的类型
> class unordered_map;
```

**1. 构造**

```c++
unordered_map<string, string> umap;
unordered_map<string, string> umap{{"key1", "value1"}, {"key2", "value2"}};
unordered_map<string, string> umap(anotherumap);
// others omitted
```

**2. 成员函数**

```c++
umap.begin();
umap.end();
umap.rbegin();
umap.rend();
umap.size();
umap.resize();
umap.empty();
umap.find(key);         // 查找位置
umap.count(key);        // 键为key的键值对数量
umap.equal_range(key);  // 返回一个pair对象，其包含2个迭代器，用于表明当前容器中键为key的键值对所在的范围
umap.insert();
umap.emplace();         // 原地指定位置插入
umap.erase();
umap.clear();
umap.swap();
umap.reverse();         // 容器内元素反转顺序
umap.hash_function();   // 哈希函数对象
```

### V.VIII. 并查集

```c++
int fa[MAXN];

void initialize(int n) {
    for (int i = 1; i <= n; ++i) {
        fa[i] = i;
    }
}

// 无路径压缩
int find(int x) {
    if (fa[x] == x) return x;
    return find(fa[x]);
}

// 路径压缩
int find(int x) {
    return fa[x] == x ? x : (fa[x] = find(fa[x]));
}

void merge(int dst, int src) {
    fa[find(dst)] = find(src);
}
```

---

## Reference

排名不分先后。

1. https://blog.csdn.net/xyqqwer/article/details/81433429
2. https://zhuanlan.zhihu.com/p/122413160
3. https://blog.csdn.net/MikeJackSTG/article/details/81806120
4. https://zhuanlan.zhihu.com/p/93857890
5. https://zhuanlan.zhihu.com/p/83334559
6. http://c.biancheng.net/stl/
7. https://www.cnblogs.com/yalphait/articles/8889221.html

---

## Changelog

### v1.0.0

**Add:**

- 补全IV.II.、IV.III.和V.；
- 补全引用；
- 增加C++ STL部分章节。

**Remove:**

- 删除线段树章节。

**Modify & Fix:**

- 修正部分标题序号；
- 修正部分用词失当。