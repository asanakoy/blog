---
layout: post
comments: true
title: "Binary Tree Right Side View"
date: 2020-04-20
tags: leetcode algorithms
thumbnail: "/assets/images/leetcode/binary-tree-right-side-view-thumbnail.png"
---

> Imagine yourself standing on the right side of it. Now return the values of the nodes you can see ordered from top to bottom.

<!--more-->
Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

Example:
```
Input: [1,2,3,null,5,null,4]
Output: [1, 3, 4]
Explanation:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
```



We use **postorder** tree traversal. 
* We maintain the current depth in the recurrent calls;
* Use vector `vals` to store rightmost value at depth `i`;
*  The nodes at the same depth will be visited from left to right. The key to ensure this is to traverse left subtree *before* the right one. 
This will allow to overwrite the value at the same depth in `vals`;
* Is is easy to see that preoreder or inorder traversal would work as well. 

Code:
{% highlight c++ %}
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    vector<int> vals;
    
    void f(TreeNode* x ,int depth) {
        if (!x)
            return;
        ++depth;
        f(x->left, depth);
        f(x->right, depth);
        if (depth  >= vals.size())
            vals.resize(depth + 1);
        vals[depth] = x->val;
    }
    
    vector<int> rightSideView(TreeNode* root) {
        f(root, -1);
        return vals;
    }
};
{% endhighlight %}


----- 

Feel free to ask me any questions in the comments below. Feedback is also very appreciated.  

- **Join** my telegram channel <img style="display:inline" src="{{ '/assets/images/telegram.png' | relative_url }}"> [@gradientdude](https://t.me/gradientdude)
- **Follow** me on twitter <img style="display:inline; height:32px" src="{{ '/assets/images/twitter.png' | relative_url }}"> [@artsiom_s](https://twitter.com/artsiom_s)

