
-- lets test our model by use cases uing following query :) 
  --- created and adjust by keyvan 2025-08-20



use LesMills_Reporting
go 
select att.*,
       crm.ClubName,
       crm.MembershipStatusReason,
	   crm.MembershipOrigin,
	   crm.MembershipStartDate,
	   crm.MembershipEndDate,
	   coalesce(s.SuspensionStatus,'No Pause') as Status,
	   r.[Rate Increase] as RateIncrease,
	   r.DateRateIncrease as DateIncreaseRate,
	   mx.FeltValued,
	   mx.FeltWelcomed,
	   mx.LikelihoodToRecommend as NPS
from pbi.LMNZ_MemberWeeklyAttendanceCount att
left join repo.CRM_ActiveMemberships crm 
on att.MembershipID=crm.MembershipID
left join repo.MSReport_List_Suspensions s
on crm.MembershipID=s.MembershipId
left join mart.lmnz_RateIncreases r
on crm.LMID=r.MemberID
left join pbi.MXMData mx
on r.MemberID=mx.MemberId
where crm.MembershipID = '0DF63A9F-F4F9-EF11-BAE2-000D3AE17142'
order by WeekBeginningDate desc


