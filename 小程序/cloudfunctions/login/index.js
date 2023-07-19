// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init()
const db = cloud.database();
const _ = db.command

// 云函数入口函数
exports.main = async (event, context) => {
  var userInfo = event.userInfo
  const wxcontext = cloud.getWXContext()
  userInfo._openid = wxcontext.OPENID;

  //promise处理异步
  return await new Promise((resolve) => {
    db.collection('defaultUserInfo').where({
        _openid: _.eq(userInfo._openid)
      }).get()
      .then(res => {
        if (res.data.length > 0) {
          resolve(res);
        } else {
          db.collection('defaultUserInfo').add({
            data: {
              nickName: userInfo.nickName,
              avatarUrl: userInfo.avatarUrl,
              _openid: wxcontext.OPENID,

            }
          }).then(res=>{
            db.collection('defaultUserInfo').where({
              _openid: _.eq(userInfo._openid)
            }).get()
            .then(res => {
              resolve(res);
            })
          })
        }
      })
  })
}
