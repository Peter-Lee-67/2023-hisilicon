// pages/cover/cover.js
var that;

Page({
  data: {
    userInfo: null,
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    that = this;
    if (getApp().globalData.userInfo) {
      wx.navigateTo({
        url: '/pages/login/login',
      })
      that.setData({
        userInfo: getApp().globalData.userInfo
      })
    }
  },

  getUserProfile(){
    wx.getUserProfile({
      desc: '请填写个人信息',
      success:(res)=>{
        console.log(res.userInfo)
        if(res.userInfo){
            that.addUser(res.userInfo)
        }
       else{
            wx.showToast({title:'拒绝授权',})
        }
      }
    })
  },

  addUser(userInfo){                                                                                     
    wx.showLoading({
    title:'正在登录', 
    })
   console.log(userInfo)
    wx.cloud.callFunction({
      name:'login',
      data:{userInfo}
    }).then(res=>{
      console.log(res);
      this.setData({
        userInfo:res.result.data[0]
      })
      console.log(that.data.userInfo)
      wx.setStorage({
        data: JSON.stringify(res.result.data[0]),
        key:'userInfo',
        success(res){
          getApp().globalData.userInfo=userInfo;
          wx.hideLoading() 
          wx.navigateTo({
        url: '/pages/login/login',
      })
        }
      })
    })
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})